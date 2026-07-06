"""Plotting helpers: turn a set of run dicts into load-sweep figures.

Headless (Agg backend) so it runs over SSH with no display. Each figure plots a metric against
arrival rate λ, averaging repeats at each λ and drawing ±1 std error bars so run-to-run variance
is visible (measurement rigor is the whole point of this phase).
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _group_by_rate(run_dicts: list[dict]) -> dict[float, list[dict]]:
    g: dict[float, list[dict]] = defaultdict(list)
    for d in run_dicts:
        g[float(d["rate"])].append(d)
    return dict(sorted(g.items()))


def _series(grouped: dict[float, list[dict]], getter):
    rates, means, stds = [], [], []
    for rate, runs in grouped.items():
        vals = []
        for d in runs:
            v = getter(d)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                vals.append(v)
        if not vals:
            continue
        rates.append(rate)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    return rates, means, stds


def _errbar(ax, grouped, getter, label):
    x, y, e = _series(grouped, getter)
    if x:
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=label)


def make_figures(run_dicts: list[dict], out_dir: str | Path, prefix: str = "phase1") -> list[Path]:
    """Write latency-, throughput-, and utilization-vs-load PNGs. Returns the file paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = _group_by_rate(run_dicts)
    paths: list[Path] = []

    # 1) Latency vs load (TTFT + TPOT, p50 & p99).
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    _errbar(a1, grouped, lambda d: d["ttft_ms"]["p50"], "TTFT p50")
    _errbar(a1, grouped, lambda d: d["ttft_ms"]["p99"], "TTFT p99")
    a1.set_xlabel("arrival rate λ (req/s)"); a1.set_ylabel("TTFT (ms)")
    a1.set_title("Time-to-first-token vs load"); a1.legend(); a1.grid(True, alpha=0.3)
    _errbar(a2, grouped, lambda d: d["tpot_ms"]["p50"], "TPOT p50")
    _errbar(a2, grouped, lambda d: d["tpot_ms"]["p99"], "TPOT p99")
    a2.set_xlabel("arrival rate λ (req/s)"); a2.set_ylabel("TPOT (ms)")
    a2.set_title("Time-per-output-token vs load"); a2.legend(); a2.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"{prefix}_latency_vs_load.png"; fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # 2) Throughput vs load.
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _errbar(ax, grouped, lambda d: d["throughput_tok_s"], "throughput")
    ax.set_xlabel("arrival rate λ (req/s)"); ax.set_ylabel("throughput (tok/s)")
    ax.set_title("Throughput vs load"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"{prefix}_throughput_vs_load.png"; fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # 3) GPU utilization vs load (SM-active + DRAM-active from DCGM).
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _errbar(ax, grouped, lambda d: d["sm_active_pct"], "SM-active %")
    _errbar(ax, grouped, lambda d: d["dram_active_pct"], "DRAM-active %")
    ax.set_xlabel("arrival rate λ (req/s)"); ax.set_ylabel("DCGM active (%)")
    ax.set_title("GPU utilization vs load"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    p = out_dir / f"{prefix}_utilization_vs_load.png"; fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    return paths
