"""Phase-1 harness runner.

Sweeps ``workload.arrival_rates`` x ``workload.repeats`` open-loop Poisson runs against a running
vLLM server, writing one results JSON per run, a per-rate variance summary, and load-sweep figures.

Run on the serving box (venv active), server already up with the n-gram spec-decode config:
    python phase1-harness/run.py --config configs/phase1_mixed.yaml

(The folder name has a hyphen so ``-m phase1.run`` won't import; run the file path directly.)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config  # noqa: E402
from common.loadgen import run_load  # noqa: E402
from common.metrics import aggregate_run, summarize_repeats, write_results  # noqa: E402
from common.plots import make_figures  # noqa: E402
from common.telemetry import TelemetrySampler  # noqa: E402


async def wait_for_health(base_url: str, timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                if (await client.get(f"{base_url}/health")).status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(1.0)
    raise TimeoutError(f"server at {base_url} not healthy within {timeout_s}s")


def _dump_raw(cfg, results, sampler, rate, repeat, results_dir: Path) -> None:
    """Optional: persist per-request records + raw telemetry time-series for later analysis."""
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg.get("_config_name"),
        "rate": rate,
        "repeat": repeat,
        "requests": [asdict(r) for r in results],
        "metrics_samples": [
            {"t": s.t, "kv": s.kv_occupancy, "running": s.num_running, "waiting": s.num_waiting,
             "power_w": s.power_w, "spec_acc": s.spec_accepted, "spec_draft": s.spec_draft}
            for s in sampler.metrics_samples
        ],
        "dcgm_samples": [{"t": s.t, "sm": s.sm_active, "dram": s.dram_active} for s in sampler.dcgm_samples],
    }
    path = raw_dir / f"{ts}_{cfg.get('_config_name')}_r{rate:g}_rep{repeat}_raw.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


async def main_async(cfg, results_dir: Path) -> None:
    base_url = cfg["server"]["base_url"]
    wl = cfg["workload"]
    tel = cfg.get("telemetry", {}) or {}
    rates = [float(r) for r in wl["arrival_rates"]]
    repeats = int(wl.get("repeats", 3))
    base_seed = int(wl.get("seed", 12345))
    save_raw = bool(wl.get("save_raw", False))

    await wait_for_health(base_url)
    print(f"[run] server healthy. sweeping rates={rates} x {repeats} repeats "
          f"(duration {wl['duration_s']}s, warmup {wl['warmup_s']}s each).")

    # One-time warm-up from idle: the very first measured run otherwise catches cold caches /
    # CUDA-graph capture and reads low (observed as a ~30% SM-active dip on run 0). Discarded.
    warm_s = float(wl.get("presweep_warmup_s", min(15.0, float(wl["warmup_s"]))))
    print(f"[run] pre-sweep warmup: {warm_s:g}s at λ={rates[0]:g} (discarded) ...")
    await run_load(cfg, rates[0], base_seed, duration_override=warm_s)

    all_runs: list[dict] = []
    for rate in rates:
        per_rate: list[dict] = []
        for rep in range(repeats):
            seed = base_seed + rep
            sampler = TelemetrySampler(
                metrics_url=f"{base_url}/metrics",
                interval_s=tel.get("interval_s", 0.5),
                gpu_index=tel.get("gpu_index", 0),
            )
            sampler.start()
            results, t0 = await run_load(cfg, rate, seed)
            sampler.stop()
            agg = aggregate_run(cfg, results, sampler, t0, rate, seed, rep)
            path = write_results(agg, results_dir)
            per_rate.append(agg)
            all_runs.append(agg)
            if save_raw:
                _dump_raw(cfg, results, sampler, rate, rep, results_dir)
            print(f"[run] λ={rate:g} rep={rep}: "
                  f"n={agg['n_requests_measured']} fail={agg['n_requests_failed']} "
                  f"thru={agg['throughput_tok_s']:.1f} tok/s "
                  f"TTFT p50/p99={agg['ttft_ms']['p50']:.0f}/{agg['ttft_ms']['p99']:.0f}ms "
                  f"SM={agg['sm_active_pct']:.0f}% acc={agg['acceptance_rate']} "
                  f"lag_p99={agg['dispatch_lag_ms']['p99']:.1f}ms -> {path.name}")

        summ = summarize_repeats(per_rate)
        t = summ["throughput_tok_s"]
        print(f"[run] λ={rate:g} SUMMARY: throughput {t['mean']:.1f}±{t['std']:.1f} tok/s "
              f"over {summ['repeats']} repeats")

    # Sweep-level artifacts: variance summary + figures.
    summaries = [summarize_repeats([r for r in all_runs if r["rate"] == rate]) for rate in rates]
    summ_path = results_dir / "summaries" / f"{cfg.get('_config_name')}_sweep_summary.json"
    summ_path.parent.mkdir(parents=True, exist_ok=True)
    with summ_path.open("w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2)
    figs = make_figures(all_runs, results_dir / "figures", prefix=cfg.get("_config_name"))
    print(f"[run] wrote sweep summary {summ_path.name} and {len(figs)} figures:")
    for f in figs:
        print(f"       {f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    cfg = load_config(args.config)
    asyncio.run(main_async(cfg, Path(args.results_dir)))


if __name__ == "__main__":
    main()
