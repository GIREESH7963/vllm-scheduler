"""Phase 3 - queueing model + validation.

Offline analysis (no serving, no GPU): fit a legible analytical model to the measured Phase-1/2
runs and show where it predicts behaviour and where it breaks down.

Model: vLLM continuous batching ~= M/G/1 **processor sharing** with a
load-dependent (batched) service rate, extended with a hard concurrency capacity limit.

    throughput(N) = min(N * r0, mumax)     # linear batching regime  ->  flat compute ceiling
    tpot(N)       = max(1/r0, N / mumax)    # flat standalone  ->  linear sharing (the PS signature)

    knee  N* = mumax / r0                   # concurrency at which per-seq fair-share == standalone

  r0    single-stream output rate (tok/s/seq)   - fit from low-concurrency runs (spare compute)
  mumax aggregate compute ceiling (tok/s)       - fit from the throughput plateau
  N     concurrency = mean running sequences (measured num_running_mean)

Capacity extension: the *hard* cap is a concurrency ceiling. Two candidate ceilings compete -
compute (the knee N*) and KV cache (C_kv sequences before the KV cache is full). We derive C_kv
from the data (C_kv = N / kv_occupancy) and show which one binds on this hardware, then project
where the other would take over.

Run:  python model/model.py --config configs/phase3_model.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import load_config  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------------------- data


def load_rows(csv_path: Path) -> list[dict]:
    import csv

    rows = []
    with csv_path.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def col(rows, key) -> np.ndarray:
    def to_f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    return np.array([to_f(r.get(key)) for r in rows])


# --------------------------------------------------------------------------------------- fit


def fit_throughput(N: np.ndarray, thr: np.ndarray, r0_grid, mumax_grid) -> dict:
    """Least-squares fit of throughput(N) = min(N*r0, mumax) over a grid. Returns fitted params."""
    m = np.isfinite(N) & np.isfinite(thr)
    N, thr = N[m], thr[m]
    r0s = np.arange(*r0_grid)
    mms = np.arange(*mumax_grid)
    best = None
    for r0 in r0s:
        for mm in mms:
            err = float(np.mean((np.minimum(N * r0, mm) - thr) ** 2))
            if best is None or err < best[0]:
                best = (err, float(r0), float(mm))
    rmse, r0, mumax = best
    return {
        "r0": r0,
        "mumax": mumax,
        "n_star": mumax / r0,
        "rmse_tok_s": rmse**0.5,
        "n_points": int(m.sum()),
    }


def refine_r0_from_low_load(N, tpot, thr, low_N: float, fitted: dict) -> dict:
    """Cross-check r0 against the low-concurrency standalone token rate (a physical anchor)."""
    m = np.isfinite(N) & np.isfinite(tpot) & (N < low_N)
    if m.sum() >= 3:
        # standalone rate from the median low-load per-token latency (robust to low-lambda noise)
        r0_tpot = 1000.0 / float(np.median(tpot[m]))
        fitted["r0_from_low_load_tpot"] = r0_tpot
        fitted["low_load_tpot_ms_p50"] = float(np.median(tpot[m]))
    return fitted


# ---------------------------------------------------------------------------------- kv capacity


def kv_capacity(N, kv, min_occ: float) -> dict:
    """C_kv = concurrent sequences that would fill the KV cache, derived from the measurements.

    At measured concurrency N with kv_occupancy f (0..1), C_kv = N / f. Median over runs where
    f is above the sampler noise floor.
    """
    m = np.isfinite(N) & np.isfinite(kv) & (kv > min_occ)
    caps = N[m] / kv[m]
    return {
        "c_kv_seqs": float(np.median(caps)),
        "c_kv_p10": float(np.percentile(caps, 10)),
        "c_kv_p90": float(np.percentile(caps, 90)),
        "n_points": int(m.sum()),
    }


# ------------------------------------------------------------------------------------- figures


def fig_throughput_vs_concurrency(rows, fits, families, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "phase2_mixed": "tab:blue",
        "phase2_chat": "tab:orange",
        "phase1_mixed": "tab:green",
    }
    for fam in families:
        sel = [r for r in rows if r["config"] == fam]
        N, thr = col(sel, "nrun"), col(sel, "thr")
        ax.scatter(
            N,
            thr,
            s=28,
            alpha=0.6,
            color=colors.get(fam, "gray"),
            label=f"{fam} (measured)",
        )
        f = fits[fam]
        xs = np.linspace(0.5, max(np.nanmax(N), f["n_star"] * 1.3), 200)
        ax.plot(
            xs,
            np.minimum(xs * f["r0"], f["mumax"]),
            color=colors.get(fam, "gray"),
            lw=2,
        )
        ax.axvline(f["n_star"], color=colors.get(fam, "gray"), ls=":", lw=1)
    ax.set_xlabel("concurrency  N  (mean running sequences)")
    ax.set_ylabel("throughput (tok/s)")
    ax.set_title(
        "Throughput vs concurrency: batching ramp -> compute ceiling\n"
        "lines = min(N*r0, mumax); dotted = knee N* = mumax/r0"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def fig_tpot_vs_concurrency(rows, fits, families, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "phase2_mixed": "tab:blue",
        "phase2_chat": "tab:orange",
        "phase1_mixed": "tab:green",
    }
    for fam in families:
        sel = [r for r in rows if r["config"] == fam]
        N, tpot = col(sel, "nrun"), col(sel, "tpot50")
        ax.scatter(
            N,
            tpot,
            s=28,
            alpha=0.6,
            color=colors.get(fam, "gray"),
            label=f"{fam} (measured p50)",
        )
        f = fits[fam]
        xs = np.linspace(0.5, np.nanmax(N) * 1.05, 200)
        # PS prediction, internally consistent with the throughput fit: tpot = N / throughput(N)
        ax.plot(
            xs,
            np.maximum(1000.0 / f["r0"], xs * 1000.0 / f["mumax"]),
            color=colors.get(fam, "gray"),
            lw=2,
        )
        ax.axvline(f["n_star"], color=colors.get(fam, "gray"), ls=":", lw=1)
    ax.set_xlabel("concurrency  N  (mean running sequences)")
    ax.set_ylabel("TPOT p50 (ms/token)")
    ax.set_title(
        "Per-token latency vs concurrency: flat (standalone) -> linear (processor sharing)\n"
        "lines = max(1/r0, N/mumax)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def fig_capacity_regimes(fits, kv, baseline_fam, current_ctx, out_path: Path):
    """The differentiator plot: which ceiling binds (compute knee N* vs KV cap C_kv), and where
    KV would take over as the sequence length grows."""
    f = fits[baseline_fam]
    n_star = f["n_star"]
    c_kv = kv["c_kv_seqs"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: the two ceilings on a log scale.
    axL.bar(
        ["compute knee\nN* = mumax/r0", "KV cap\nC_kv"],
        [n_star, c_kv],
        color=["tab:red", "tab:purple"],
    )
    axL.set_yscale("log")
    axL.set_ylabel("concurrent sequences")
    axL.set_title(
        f"Binding constraint on T4 + 1.5B ({baseline_fam})\n"
        f"compute saturates ~{c_kv / n_star:.0f}x before KV"
    )
    for i, v in enumerate([n_star, c_kv]):
        axL.text(i, v * 1.1, f"{v:.0f}", ha="center", fontsize=10)

    # Right: C_kv shrinks as avg sequence length grows; KV binds where it crosses N*.
    ctx = np.linspace(current_ctx, current_ctx * 200, 400)
    c_kv_ctx = c_kv * (
        current_ctx / ctx
    )  # KV budget in tokens is fixed -> seqs ~ 1/context
    axR.plot(ctx, c_kv_ctx, color="tab:purple", lw=2, label="C_kv(context)")
    axR.axhline(
        n_star, color="tab:red", ls="--", lw=1.5, label=f"compute knee N*={n_star:.0f}"
    )
    cross = current_ctx * c_kv / n_star  # context where C_kv == N*
    axR.axvline(cross, color="k", ls=":", lw=1)
    axR.text(cross, n_star * 3, f"KV binds\n~{cross / 1000:.0f}k tok", fontsize=9)
    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlabel("avg sequence length (tokens)")
    axR.set_ylabel("concurrent sequences")
    axR.set_title("Where KV would become the binding constraint")
    axR.legend(fontsize=8)
    for a in (axL, axR):
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return {
        "kv_binds_context_tokens": float(cross),
        "compute_vs_kv_headroom_x": float(c_kv / n_star),
    }


def fig_lambda_overlay(rows, fits, families, out_path: Path):
    """Throughput vs arrival rate lambda: model = min(lambda*b, mumax), overlaid on measured means.

    Restricted to the no-scheduler runs (policy nocap or Phase-1 blank): the model is the
    admit-all baseline, so mixing in throughput-capping policies would not be apples-to-apples.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "phase2_mixed": "tab:blue",
        "phase2_chat": "tab:orange",
        "phase1_mixed": "tab:green",
    }
    for fam in families:
        sel = [
            r
            for r in rows
            if r["config"] == fam and (r.get("policy") in ("nocap", "", None))
        ]
        lam, thr, out = col(sel, "rate"), col(sel, "thr"), col(sel, "outact")
        b = float(np.nanmean(out))  # mean output tokens per request for this mix
        # measured mean throughput per lambda
        for L in sorted(set(lam[np.isfinite(lam)])):
            mm = lam == L
            ax.scatter([L], [np.nanmean(thr[mm])], s=45, color=colors.get(fam, "gray"))
        f = fits[fam]
        xs = np.linspace(0.2, np.nanmax(lam) * 1.15, 100)
        ax.plot(
            xs,
            np.minimum(xs * b, f["mumax"]),
            color=colors.get(fam, "gray"),
            lw=2,
            label=f"{fam}: min(lambda*{b:.0f}, {f['mumax']:.0f})",
        )
        ax.axvline(f["mumax"] / b, color=colors.get(fam, "gray"), ls=":", lw=1)
    ax.set_xlabel("arrival rate  lambda  (req/s)")
    ax.set_ylabel("throughput (tok/s)")
    ax.set_title(
        "Throughput vs load: rises with lambda then saturates at the compute ceiling\n"
        "dotted = saturation knee lambda* = mumax / mean_output_tokens"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------- kv-regime validation


def _context_from_name(name: str) -> float:
    """Parse nominal context tokens from a kvsweep config name, e.g. 'kvsweep_ctx16k' -> 16000."""
    import re

    m = re.search(r"ctx(\d+)k", name)
    return float(m.group(1)) * 1000.0 if m else float("nan")


def fig_kv_regime_validation(kv_rows, n_star, c_kv, ctx0, kv_bound_occ, out_path: Path) -> dict:
    """Overlay MEASURED achieved concurrency vs context on the predicted min(N*, C_kv(context))
    envelope. This is the empirical test of the derived crossover: as context grows, the binding
    ceiling should hand off from compute (flat N*) to KV (the 1/context C_kv curve), and the KV cache
    should saturate (kv_occupancy -> 1, num_waiting > 0) exactly where the two cross.
    """
    # group by config -> (context, mean/std of achieved concurrency, kv occupancy, waiting)
    by_ctx = {}
    for r in kv_rows:
        ctx = _context_from_name(r.get("config", ""))
        if not np.isfinite(ctx):
            continue
        by_ctx.setdefault(ctx, {"nrun": [], "kv": [], "nwait": []})
        for k, col_name in (("nrun", "nrun"), ("kv", "kv"), ("nwait", "nwait")):
            try:
                by_ctx[ctx][k].append(float(r.get(col_name)))
            except (TypeError, ValueError):
                pass
    if not by_ctx:
        return {}
    ctxs = np.array(sorted(by_ctx))
    nrun_mean = np.array([np.nanmean(by_ctx[c]["nrun"]) for c in ctxs])
    nrun_std = np.array([np.nanstd(by_ctx[c]["nrun"]) for c in ctxs])
    kv_mean = np.array([np.nanmean(by_ctx[c]["kv"]) for c in ctxs])
    nwait_mean = np.array([np.nanmean(by_ctx[c]["nwait"]) for c in ctxs])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    xs = np.linspace(ctxs.min() * 0.8, ctxs.max() * 1.15, 300)
    c_kv_curve = c_kv * (ctx0 / xs)  # KV cap shrinks as 1/context (fixed token budget)
    ax.plot(xs, np.minimum(n_star, c_kv_curve), "k-", lw=2,
            label="predicted ceiling  min(N*, C_kv(context))")
    ax.axhline(n_star, color="tab:red", ls="--", lw=1.2, label=f"compute knee N*={n_star:.0f}")
    ax.plot(xs, c_kv_curve, color="tab:purple", ls=":", lw=1.2, label="KV cap C_kv(context)")

    # measured achieved concurrency; color points by regime (KV-bound if kv high or waiting > 0)
    kv_bound = (kv_mean >= kv_bound_occ) | (nwait_mean > 0.5)
    ax.errorbar(ctxs[~kv_bound], nrun_mean[~kv_bound], yerr=nrun_std[~kv_bound], fmt="o",
                color="tab:blue", ms=8, capsize=4, label="measured (compute-bound)")
    ax.errorbar(ctxs[kv_bound], nrun_mean[kv_bound], yerr=nrun_std[kv_bound], fmt="s",
                color="tab:purple", ms=9, capsize=4, label="measured (KV-bound: kv~1 / queueing)")
    for c, n, k in zip(ctxs, nrun_mean, kv_mean):
        ax.annotate(f"kv={k:.2f}", (c, n), textcoords="offset points", xytext=(6, 6), fontsize=8)

    cross = ctx0 * c_kv / n_star
    ax.axvline(cross, color="gray", ls=":", lw=1)
    ax.text(cross, n_star * 1.15, f"predicted\ncrossover\n~{cross / 1000:.0f}k", fontsize=8, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("prompt context (tokens)")
    ax.set_ylabel("achieved concurrency  N  (mean running sequences)")
    ax.set_title("KV-regime validation: does achieved concurrency follow min(N*, C_kv(context))?\n"
                 "compute-bound (flat at N*) hands off to KV-bound (1/context) as prompts grow")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return {
        "contexts": ctxs.tolist(),
        "achieved_concurrency_mean": nrun_mean.tolist(),
        "kv_occupancy_mean": kv_mean.tolist(),
        "num_waiting_mean": nwait_mean.tolist(),
        "kv_bound_regime": kv_bound.tolist(),
        "predicted_crossover_tokens": float(cross),
    }


# ---------------------------------------------------------------------------------- divergence


def divergence_report(rows, fits, families, rel_flag: float) -> list[dict]:
    """Where measured throughput departs from the model by more than rel_flag (fractional)."""
    flags = []
    for fam in families:
        sel = [r for r in rows if r["config"] == fam]
        N, thr = col(sel, "nrun"), col(sel, "thr")
        f = fits[fam]
        pred = np.minimum(N * f["r0"], f["mumax"])
        rel = (thr - pred) / pred
        m = np.isfinite(rel) & (np.abs(rel) > rel_flag)
        for i in np.where(m)[0]:
            flags.append(
                {
                    "family": fam,
                    "N": float(N[i]),
                    "measured_tok_s": float(thr[i]),
                    "model_tok_s": float(pred[i]),
                    "rel_error": float(rel[i]),
                }
            )
    flags.sort(key=lambda d: abs(d["rel_error"]), reverse=True)
    return flags


# --------------------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    root = Path(__file__).resolve().parent.parent
    rows = load_rows(root / cfg["data"]["csv"])
    fitc = cfg["fit"]
    families = fitc["families"]

    # ---- fit each workload family ----
    fits = {}
    for fam in families:
        sel = [r for r in rows if r["config"] == fam]
        N, thr, tpot = col(sel, "nrun"), col(sel, "thr"), col(sel, "tpot50")
        f = fit_throughput(N, thr, fitc["r0_grid"], fitc["mumax_grid"])
        f = refine_r0_from_low_load(N, tpot, thr, fitc["low_concurrency_N"], f)
        fits[fam] = f

    # ---- KV capacity from the data ----
    Nall, kvall = col(rows, "nrun"), col(rows, "kv")
    kv = kv_capacity(Nall, kvall, cfg["kv"]["min_kv_occupancy"])

    # ---- figures ----
    figdir = root / cfg["output"]["figures_dir"]
    figdir.mkdir(parents=True, exist_ok=True)
    fig_throughput_vs_concurrency(
        rows, fits, families, figdir / "phase3_throughput_vs_concurrency.png"
    )
    fig_tpot_vs_concurrency(
        rows, fits, families, figdir / "phase3_tpot_vs_concurrency.png"
    )
    fig_lambda_overlay(rows, fits, families, figdir / "phase3_throughput_vs_load.png")
    baseline_fam = fitc["baseline_family"]
    kv_proj = fig_capacity_regimes(
        fits,
        kv,
        baseline_fam,
        cfg["kv"]["current_avg_context_tokens"],
        figdir / "phase3_capacity_regimes.png",
    )

    # ---- divergence ----
    flags = divergence_report(rows, fits, families, cfg["divergence"]["rel_error_flag"])

    # ---- KV-regime validation (optional; only if the long-context sweep has been run) ----
    kv_regime = {}
    krc = cfg.get("kv_regime") or {}
    kv_csv = root / krc.get("csv", "model/kvsweep_runs.csv") if krc else None
    if kv_csv is not None and kv_csv.exists():
        kv_rows = load_rows(kv_csv)
        kv_regime = fig_kv_regime_validation(
            kv_rows, fits[baseline_fam]["n_star"], kv["c_kv_seqs"],
            cfg["kv"]["current_avg_context_tokens"], krc.get("kv_bound_occupancy", 0.85),
            figdir / "phase3_kv_regime_validation.png",
        )
        print(f"  KV-regime overlay: {len(kv_regime.get('contexts', []))} context points "
              f"-> phase3_kv_regime_validation.png")
    else:
        print("  KV-regime sweep not present yet (model/kvsweep_runs.csv) — skipping overlay")

    # ---- write summary ----
    summary = {
        "model": "M/G/1-PS with batched service rate + concurrency capacity cap",
        "fits": fits,
        "kv_capacity": {**kv, **kv_proj},
        "baseline": {"family": baseline_fam, **fits[baseline_fam]},
        "divergence_points": flags,
        "n_divergence_points": len(flags),
        "kv_regime_validation": kv_regime,
    }
    out_json = root / cfg["output"]["summary_json"]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # ---- console report ----
    print(f"== Phase-3 model fit ({len(rows)} measured runs) ==")
    for fam, f in fits.items():
        print(
            f"  {fam:14s} r0={f['r0']:.1f} tok/s/seq  mumax={f['mumax']:.0f} tok/s  "
            f"N*={f['n_star']:.1f}  rmse={f['rmse_tok_s']:.1f} (n={f['n_points']})"
        )
    b = fits[baseline_fam]
    print(
        f"  baseline ({baseline_fam}): knee N*={b['n_star']:.1f}, "
        f"standalone TPOT p50={b.get('low_load_tpot_ms_p50', float('nan')):.1f}ms "
        f"-> r0~{b.get('r0_from_low_load_tpot', float('nan')):.0f} tok/s (anchor)"
    )
    print(
        f"  KV cap C_kv={kv['c_kv_seqs']:.0f} seqs  -> compute saturates "
        f"{kv_proj['compute_vs_kv_headroom_x']:.0f}x before KV; "
        f"KV would bind at ~{kv_proj['kv_binds_context_tokens'] / 1000:.0f}k-token sequences"
    )
    print(
        f"  {len(flags)} divergence points (>|{cfg['divergence']['rel_error_flag'] * 100:.0f}%|)"
    )
    print(
        f"  wrote {out_json.relative_to(root)} + 4 figures to {figdir.relative_to(root)}/"
    )


if __name__ == "__main__":
    main()
