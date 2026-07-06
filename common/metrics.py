"""Aggregate per-request timings + telemetry into the project's results schema.

Schema (see README) — one JSON per run:
  config, throughput_tok_s, ttft_ms{p50,p99}, tpot_ms{p50,p99}, kv_occupancy,
  sm_active_pct, dram_active_pct, energy_j_per_tok, slo_attainment, acceptance_rate

Convention: always report p50 AND p99, never just the mean.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .telemetry import TelemetrySampler


@dataclass
class RequestRecord:
    """One measured (post-warmup) request, timed client-side."""

    ttft_ms: float
    tpot_ms: float  # mean per-output-token latency for this request
    output_tokens: int
    success: bool


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


def _p50_p99(values: list[float]) -> dict[str, float]:
    return {"p50": _pct(values, 50), "p99": _pct(values, 99)}


def _nanmean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


def build_results(
    config,
    records: list[RequestRecord],
    sampler: TelemetrySampler,
    wall_time_s: float,
) -> dict:
    """Assemble the results dict from client timings + collected telemetry."""
    ok = [r for r in records if r.success]
    ttfts = [r.ttft_ms for r in ok]
    tpots = [r.tpot_ms for r in ok if not math.isnan(r.tpot_ms)]
    total_out_tokens = sum(r.output_tokens for r in ok)

    # Throughput over the measured wall-clock window.
    throughput = total_out_tokens / wall_time_s if wall_time_s > 0 else float("nan")

    # KV occupancy + power from the /metrics+NVML poll stream.
    kv = _nanmean([s.kv_occupancy for s in sampler.metrics_samples])
    mean_power_w = _nanmean([s.power_w for s in sampler.metrics_samples])

    # SM/DRAM active from DCGM (0..1 -> percent).
    sm = _nanmean([s.sm_active for s in sampler.dcgm_samples]) * 100.0
    dram = _nanmean([s.dram_active for s in sampler.dcgm_samples]) * 100.0

    # Energy per token: mean board power (J/s) * window (s) / tokens.
    energy_j_per_tok = (
        mean_power_w * wall_time_s / total_out_tokens
        if total_out_tokens > 0 and not math.isnan(mean_power_w)
        else float("nan")
    )

    # SLO attainment: fraction of measured requests meeting both TTFT and TPOT thresholds.
    slo = config.get("slo", {}) or {}
    ttft_slo = slo.get("ttft_ms")
    tpot_slo = slo.get("tpot_ms")
    if ok and (ttft_slo is not None or tpot_slo is not None):
        met = sum(
            1
            for r in ok
            if (ttft_slo is None or r.ttft_ms <= ttft_slo)
            and (tpot_slo is None or math.isnan(r.tpot_ms) or r.tpot_ms <= tpot_slo)
        )
        slo_attainment = met / len(ok)
    else:
        slo_attainment = float("nan")

    # Acceptance rate from spec-decode counters (delta across the run), if exposed.
    acceptance_rate = _acceptance_rate(sampler)

    return {
        "config": config.get("_config_name", "unknown"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config.get("model"),
        "n_requests_measured": len(ok),
        "n_requests_failed": len(records) - len(ok),
        "throughput_tok_s": throughput,
        "ttft_ms": _p50_p99(ttfts),
        "tpot_ms": _p50_p99(tpots),
        "kv_occupancy": kv,
        "sm_active_pct": sm,
        "dram_active_pct": dram,
        "mean_power_w": mean_power_w,
        "energy_j_per_tok": energy_j_per_tok,
        "slo_attainment": slo_attainment,
        "acceptance_rate": acceptance_rate,
        "notes": sampler.notes,
    }


def _acceptance_rate(sampler: TelemetrySampler) -> float | None:
    acc = [s.spec_accepted for s in sampler.metrics_samples if s.spec_accepted is not None]
    draft = [s.spec_draft for s in sampler.metrics_samples if s.spec_draft is not None]
    if not acc or not draft:
        return None
    d_acc = acc[-1] - acc[0]
    d_draft = draft[-1] - draft[0]
    if d_draft <= 0:
        return None
    return d_acc / d_draft


def write_results(results: dict, results_dir: str | Path = "results") -> Path:
    """Write one results JSON to results/<timestamp>_<config>.json and return its path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = ""
    if results.get("rate") is not None:
        suffix += f"_r{results['rate']:g}"
    if results.get("repeat") is not None:
        suffix += f"_rep{results['repeat']}"
    path = results_dir / f"{ts}_{results.get('config', 'run')}{suffix}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return path


# =========================================================================================
# Phase 1: open-loop aggregation (warmup filtering, per-class breakdown, repeat variance)
# =========================================================================================


def _win(samples, t_start):
    return [s for s in samples if s.t >= t_start]


def aggregate_run(config, results, sampler, t0: float, rate: float, seed: int, repeat: int) -> dict:
    """Aggregate one open-loop run to the root schema + Phase-1 extras.

    ``results`` are ``loadgen.RequestResult``; warmup requests (``is_warmup``) are discarded from
    every statistic. Telemetry is filtered to the post-warmup window using the shared monotonic
    clock (``t0`` is the run start; warmup ends at ``t0 + warmup_s``).
    """
    wl = config.get("workload", {}) or {}
    duration_s = float(wl.get("duration_s", 120))
    warmup_s = float(wl.get("warmup_s", 30))
    measure_window_s = max(1e-9, duration_s - warmup_s)
    t_measure_start = t0 + warmup_s

    measured = [r for r in results if not r.is_warmup]
    ok = [r for r in measured if r.success]
    ttfts = [r.ttft_ms for r in ok if not math.isnan(r.ttft_ms)]
    tpots = [r.tpot_ms for r in ok if not math.isnan(r.tpot_ms)]
    e2e = [r.latency_ms for r in ok if not math.isnan(r.latency_ms)]
    # Phase-2: queue wait (arrival->admission) and total arrival->last-token latency.
    qwaits = [r.queue_wait_ms for r in ok]
    totals = [r.queue_wait_ms + r.latency_ms for r in ok if not math.isnan(r.latency_ms)]
    total_out = sum(r.output_tokens_actual for r in ok)
    throughput = total_out / measure_window_s

    msamps = _win(sampler.metrics_samples, t_measure_start)
    dsamps = _win(sampler.dcgm_samples, t_measure_start)
    kv = _nanmean([s.kv_occupancy for s in msamps])
    running = _nanmean([s.num_running for s in msamps])
    waiting = _nanmean([s.num_waiting for s in msamps])
    mean_power_w = _nanmean([s.power_w for s in msamps])
    sm = _nanmean([s.sm_active for s in dsamps]) * 100.0
    dram = _nanmean([s.dram_active for s in dsamps]) * 100.0
    # Thermal validity: on the passively-cooled T4, clock throttling silently corrupts timings.
    temps = [s.temp_c for s in msamps if not math.isnan(s.temp_c)]
    max_temp_c = float(max(temps)) if temps else float("nan")
    thermal_throttled = any(s.throttled for s in msamps)
    energy_j_per_tok = (
        mean_power_w * measure_window_s / total_out
        if total_out > 0 and not math.isnan(mean_power_w)
        else float("nan")
    )

    slo = config.get("slo", {}) or {}
    ttft_slo, tpot_slo = slo.get("ttft_ms"), slo.get("tpot_ms")
    if measured and (ttft_slo is not None or tpot_slo is not None):
        # Denominator is ALL measured requests: a dropped/failed request is an SLO miss, not excluded
        # (else load-shedding policies would look artificially good). Effective TTFT is
        # arrival-relative: queue wait (0 in direct Phase-1 runs) + server TTFT.
        met = sum(
            1
            for r in measured
            if r.success
            and (ttft_slo is None or (r.queue_wait_ms + r.ttft_ms) <= ttft_slo)
            and (tpot_slo is None or math.isnan(r.tpot_ms) or r.tpot_ms <= tpot_slo)
        )
        slo_attainment = met / len(measured)
    else:
        slo_attainment = float("nan")

    # Client-health guard: scheduled-arrival -> actual-dispatch lag. Large lag => client is the
    # bottleneck (CPU-bound at high λ) and the open-loop assumption is violated.
    lags_ms = [(r.dispatch_t - r.arrival_t) * 1000.0 for r in measured]

    return {
        "config": config.get("_config_name", "unknown"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config.get("model"),
        "rate": rate,
        "seed": seed,
        "repeat": repeat,
        "arrivals_total": len(results),
        "n_requests_measured": len(ok),
        "n_requests_failed": len(measured) - len(ok),
        "throughput_tok_s": throughput,
        "ttft_ms": _p50_p99(ttfts),
        "tpot_ms": _p50_p99(tpots),
        "e2e_latency_ms": _p50_p99(e2e),
        "queue_wait_ms": _p50_p99(qwaits),
        "total_latency_ms": _p50_p99(totals),
        "kv_occupancy": kv,
        "num_running_mean": running,
        "num_waiting_mean": waiting,
        "sm_active_pct": sm,
        "dram_active_pct": dram,
        "mean_power_w": mean_power_w,
        "energy_j_per_tok": energy_j_per_tok,
        "max_temp_c": max_temp_c,
        "thermal_throttled": thermal_throttled,  # True => timings suspect (clocks were slowed)
        "slo_attainment": slo_attainment,
        "acceptance_rate": _acceptance_rate_windowed(msamps),
        "dispatch_lag_ms": {
            "p50": _pct(lags_ms, 50),
            "p99": _pct(lags_ms, 99),
            "max": max(lags_ms) if lags_ms else float("nan"),
        },
        "output_len": _len_stats(ok),
        "by_class": _by_class(measured, ttft_slo, tpot_slo),
        "notes": sampler.notes,
    }


def _acceptance_rate_windowed(msamps) -> float | None:
    acc = [s.spec_accepted for s in msamps if s.spec_accepted is not None]
    draft = [s.spec_draft for s in msamps if s.spec_draft is not None]
    if len(acc) < 2 or len(draft) < 2:
        return None
    d_acc, d_draft = acc[-1] - acc[0], draft[-1] - draft[0]
    return d_acc / d_draft if d_draft > 0 else None


def _len_stats(ok) -> dict:
    """Actual vs requested output length — Phase-2 needs the discrepancy."""
    req = [r.output_tokens_requested for r in ok]
    act = [r.output_tokens_actual for r in ok]
    return {
        "requested_mean": float(np.mean(req)) if req else float("nan"),
        "actual_mean": float(np.mean(act)) if act else float("nan"),
        "actual_p50": _pct([float(x) for x in act], 50),
        "actual_p99": _pct([float(x) for x in act], 99),
        "actual_over_requested": (float(np.sum(act)) / float(np.sum(req))) if np.sum(req) else float("nan"),
    }


def _by_class(measured, ttft_slo, tpot_slo) -> dict:
    """Per-class breakdown incl. queue wait, total latency, and SLO — the differentiated-service
    lens: under overload, ordering policies protect some classes at the expense of others."""
    classes = sorted({r.profile for r in measured})
    out = {}
    for c in classes:
        mc = [r for r in measured if r.profile == c]           # all measured of this class
        rc = [r for r in mc if r.success]                       # successful ones
        totals = [r.queue_wait_ms + r.latency_ms for r in rc if not math.isnan(r.latency_ms)]
        if mc and (ttft_slo is not None or tpot_slo is not None):
            met = sum(
                1 for r in mc
                if r.success
                and (ttft_slo is None or (r.queue_wait_ms + r.ttft_ms) <= ttft_slo)
                and (tpot_slo is None or math.isnan(r.tpot_ms) or r.tpot_ms <= tpot_slo)
            )
            slo_c = met / len(mc)
        else:
            slo_c = float("nan")
        out[c] = {
            "spec_decode": bool(rc[0].spec_decode) if rc else False,
            "n": len(rc),
            "n_dropped": len(mc) - len(rc),
            "ttft_ms": _p50_p99([r.ttft_ms for r in rc if not math.isnan(r.ttft_ms)]),
            "tpot_ms": _p50_p99([r.tpot_ms for r in rc if not math.isnan(r.tpot_ms)]),
            "queue_wait_ms": _p50_p99([r.queue_wait_ms for r in rc]),
            "total_latency_ms": _p50_p99(totals),
            "slo_attainment": slo_c,
            "output_tokens_actual_mean": float(np.mean([r.output_tokens_actual for r in rc])) if rc else float("nan"),
        }
    return out


def summarize_repeats(run_dicts: list[dict]) -> dict:
    """Given aggregated dicts for the SAME rate across repeats, report mean±std of key scalars."""
    if not run_dicts:
        return {}

    def col(getter):
        vals = [getter(d) for d in run_dicts]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), "n": len(vals)}

    return {
        "config": run_dicts[0].get("config"),
        "rate": run_dicts[0].get("rate"),
        "repeats": len(run_dicts),
        "throughput_tok_s": col(lambda d: d["throughput_tok_s"]),
        "ttft_ms_p50": col(lambda d: d["ttft_ms"]["p50"]),
        "ttft_ms_p99": col(lambda d: d["ttft_ms"]["p99"]),
        "tpot_ms_p50": col(lambda d: d["tpot_ms"]["p50"]),
        "tpot_ms_p99": col(lambda d: d["tpot_ms"]["p99"]),
        "sm_active_pct": col(lambda d: d["sm_active_pct"]),
        "acceptance_rate": col(lambda d: d.get("acceptance_rate")),
        "slo_attainment": col(lambda d: d["slo_attainment"]),
    }
