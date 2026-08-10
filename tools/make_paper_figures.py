"""Generate the paper figure set.

    python tools/make_paper_figures.py --out results/figures/paper

Each figure is written as PNG (review) and PDF (vector, for the manuscript). Figures that
depend on an experiment that has not finished are skipped with a message rather than drawn from
partial data.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_style as ps  # noqa: E402
from common.stats import describe  # noqa: E402

ps.apply()


# ======================================================================================
# data loading
# ======================================================================================

def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and "num_running_mean" in d:
            d["_file"] = p.name
            runs.append(d)
    return runs


def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ======================================================================================
# fig 1 — system architecture
# ======================================================================================

def fig_architecture(out: Path):
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, title, sub, face="#ffffff", edge=ps.MUTED, tc=ps.INK, lw=1.1):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                    fc=face, ec=edge, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="top",
                fontsize=9.2, fontweight="bold", color=tc, zorder=3)
        if sub:
            ax.text(x + w / 2, y + h - 8.2, sub, ha="center", va="top",
                    fontsize=7.4, color=ps.INK_2, zorder=3, linespacing=1.45)

    def arrow(x1, y1, x2, y2, color=ps.MUTED, style="-|>", lw=1.2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                                     lw=lw, linestyle=ls, mutation_scale=11,
                                     shrinkA=1, shrinkB=1, zorder=1))

    # Load generation
    box(1, 27, 20, 21, "Workload generator",
        "Poisson(λ) arrivals\nfixed seed · open loop\nclasses: short · coding\nreasoning · long",
        face="#f7f9fc", edge=ps.SLOTS[0])

    # Scheduling layer — the contribution
    box(26.5, 27, 22, 21, "Admission / ordering layer",
        "FCFS · SRPT · EDF · adaptive\nadmission cap N_adm\n\nORDER + ADMIT ONLY\n(no vLLM internals)",
        face="#fff6f2", edge=ps.SLOTS[1], lw=1.6)

    # Engine
    box(54, 27, 24, 21, "vLLM engine  (V0)",
        "continuous batching\nmax_num_seqs = 256\nn-gram spec decode, k=4\npaged KV cache", face="#f4fbf8", edge=ps.SLOTS[2])

    # GPU
    box(83, 27, 16, 21, "Tesla T4",
        "14.56 GiB usable\n40 SMs · 320 GB/s\npassively cooled", face="#fffdf5", edge=ps.SLOTS[3])

    for x1, x2 in ((21, 26.5), (48.5, 54), (78, 83)):
        arrow(x1, 37.5, x2, 37.5)

    # Telemetry plane
    box(26.5, 4, 52, 14, "Telemetry plane",
        "vLLM /metrics  →  KV occupancy, num_running, num_waiting\n"
        "DCGM  →  SM-active, DRAM-active     ·     NVML  →  power, temperature, memory",
        face="#fafafa", edge=ps.MUTED)

    arrow(70, 27, 70, 18, color=ps.SLOTS[2], ls=(0, (4, 2)))
    arrow(91, 27, 78.5, 18, color=ps.SLOTS[3], ls=(0, (4, 2)))
    # Feedback path used by the adaptive policy.
    arrow(31, 18, 31, 27, color=ps.SLOTS[1], ls=(0, (4, 2)))
    ax.text(21.5, 22.5, "feedback\n(adaptive policy)", fontsize=7, color=ps.SLOTS[1],
            va="center", ha="center", linespacing=1.4)

    # The failure this paper is about: mark the engine, caption it clear of the arrows.
    ax.add_patch(FancyBboxPatch((54, 27), 24, 21, boxstyle="round,pad=0.9,rounding_size=1.2",
                                fc="none", ec=ps.STATUS_BAD, lw=1.3, ls=(0, (4, 2)), zorder=4))
    ax.text(99, 21.5, "OOM originates inside the engine:  the speculative-decode scorer\n"
                      "allocates outside the paged KV allocator  (§ Experiment A)",
            ha="right", va="top", fontsize=7.6, color=ps.STATUS_BAD, linespacing=1.5)

    ax.set_title("System architecture: an admission layer in front of an unmodified vLLM engine",
                 pad=12, loc="left")
    return ps.save(fig, out, "fig01_architecture")


# ======================================================================================
# fig 2 — throughput vs concurrency, with the fitted model
# ======================================================================================

def fig_throughput_vs_concurrency(runs, model_fit, out: Path):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    by_cfg = defaultdict(list)
    for r in runs:
        if (r.get("n_requests_failed") or 0) == 0 and r.get("n_requests_measured"):
            by_cfg[r["config"]].append(r)

    order = [c for c in ("phase1_mixed", "modelval_mixed", "phase2_mixed", "phase2_chat")
             if c in by_cfg]
    order += [c for c in sorted(by_cfg) if c not in order]

    for i, cfg in enumerate(order[:5]):
        rs = by_cfg[cfg]
        N = np.array([r["num_running_mean"] for r in rs])
        X = np.array([r["throughput_tok_s"] for r in rs])
        ax.scatter(N, X, s=34, color=ps.SLOTS[i], marker=["o", "s", "^", "D", "v"][i],
                   edgecolor="white", linewidth=0.8, label=cfg, zorder=3, alpha=0.95)

    # Reference fit overlay.
    ref = (model_fit or {}).get("per_workload", {}).get("phase1_mixed")
    if ref and "r0_tok_s_per_seq" in ref:
        r0, mu = ref["r0_tok_s_per_seq"], ref["mu_max_tok_s"]
        xs = np.linspace(0, 150, 400)
        ax.plot(xs, np.minimum(xs * r0, mu), color=ps.INK, lw=1.6, ls=(0, (5, 2)), zorder=4,
                label=f"fit: min(N·{r0:.1f}, {mu:.0f})")
        ax.axvline(ref["N_star"], color=ps.MUTED, lw=0.9, ls=":", zorder=1)
        ps.annotate(ax, f"N* = {ref['N_star']:.1f}", (ref["N_star"], mu * 1.18),
                    (ref["N_star"] + 12, mu * 1.25))

    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Concurrent sequences in engine,  N  (num_requests_running)")
    ax.set_ylabel("Output throughput  (tok/s)")
    ax.set_title("Throughput saturates with concurrency; the knee is workload-specific", loc="left")
    ax.legend(loc="lower right", ncols=2)
    ax.text(0.012, 0.97, "fit shown for phase1_mixed only —\npooled fit is invalid (R² = 0.22)",
            transform=ax.transAxes, fontsize=7.4, color=ps.MUTED, va="top", linespacing=1.4)
    return ps.save(fig, out, "fig02_throughput_vs_concurrency")


# ======================================================================================
# fig 3 / 4 / 9 — Experiment A trajectories
# ======================================================================================

def _expA_windows(expA):
    """(trial, N, kv, mem, temp) arrays over the live portion of each trial."""
    out = []
    for rec in expA or []:
        win = rec.get("pre_failure_window") or []
        mem = np.array([s.get("mem_used_gib", np.nan) for s in win], float)
        if not np.isfinite(mem).any():
            continue
        live = mem >= np.nanmax(mem) * 0.5
        N = np.array([s.get("num_running", np.nan) for s in win], float)[live]
        kv = np.array([s.get("kv_occupancy", np.nan) for s in win], float)[live]
        t = np.array([s.get("temp_c", np.nan) for s in win], float)[live]
        out.append((rec.get("trial"), N, kv, mem[live], t))
    return out


def fig_kv_vs_concurrency(expA, runs, out: Path):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    rs = [r for r in runs if (r.get("n_requests_failed") or 0) == 0
          and isinstance(r.get("kv_occupancy"), (int, float))]
    ax.scatter([r["num_running_mean"] for r in rs], [r["kv_occupancy"] * 100 for r in rs],
               s=22, color=ps.MUTED, alpha=0.45, marker="o", linewidth=0,
               label="sweep runs (run means)", zorder=2)

    for i, (trial, N, kv, mem, t) in enumerate(_expA_windows(expA)):
        ok = np.isfinite(N) & np.isfinite(kv)
        ax.plot(N[ok], kv[ok] * 100, color=ps.SLOTS[i], lw=1.5, alpha=0.9,
                marker=["o", "s", "^"][i % 3], markersize=3, markevery=8,
                label=f"Exp A trial {trial} (to failure)", zorder=3)
        if ok.any():
            ax.scatter([N[ok][-1]], [kv[ok][-1] * 100], s=95, marker="X",
                       color=ps.STATUS_BAD, edgecolor="white", linewidth=1.1, zorder=6)

    ax.axhline(100, color=ps.MUTED, lw=0.9, ls=":")
    ax.text(248, 102, "KV cache exhausted (100%)", fontsize=7.4, color=ps.MUTED, ha="right")
    ax.axvline(256, color=ps.MUTED, lw=0.9, ls="--")
    ax.text(262, 70, "max_num_seqs\n= 256", fontsize=7.4, color=ps.MUTED, linespacing=1.4)

    ps.annotate(ax, "OOM  ×3\nKV = 28.4% ± 0.3%", (254, 28.4), (150, 52), color=ps.STATUS_BAD)
    ax.set_xlabel("Concurrent sequences,  N")
    ax.set_ylabel("KV cache occupancy  (%)")
    ax.set_ylim(-3, 112)
    ax.set_xlim(-8, 300)
    ax.set_title("The engine dies with 71.6% of its KV cache unused", loc="left", pad=10)
    # Placed low-left: the data occupies the diagonal, leaving this corner clear.
    ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.90), ncols=1)
    return ps.save(fig, out, "fig03_kv_vs_concurrency")


def fig_memory_vs_concurrency(expA, out: Path):
    wins = _expA_windows(expA)
    if not wins:
        print("  fig04 skipped — no Experiment A windows")
        return []
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    for i, (trial, N, kv, mem, t) in enumerate(wins):
        ok = np.isfinite(N) & np.isfinite(mem)
        ax.plot(N[ok], mem[ok], color=ps.SLOTS[i], lw=1.5, alpha=0.9,
                marker=["o", "s", "^"][i % 3], markersize=3, markevery=8,
                label=f"trial {trial}", zorder=3)
        if ok.any():
            ax.scatter([N[ok][-1]], [mem[ok][-1]], s=95, marker="X", color=ps.STATUS_BAD,
                       edgecolor="white", linewidth=1.1, zorder=6)

    ax.axhline(14.56, color=ps.STATUS_BAD, lw=1.0, ls="--")
    ax.text(4, 14.63, "device capacity 14.56 GiB", fontsize=7.4, color=ps.STATUS_BAD)
    ax.axhline(13.11, color=ps.MUTED, lw=0.9, ls=":")
    ax.text(4, 13.18, "vLLM budget (gpu_memory_utilization = 0.9) = 13.11 GiB",
            fontsize=7.4, color=ps.MUTED)
    ax.axvline(256, color=ps.MUTED, lw=0.9, ls="--")
    ax.text(258, 12.0, "max_num_seqs", fontsize=7.4, color=ps.MUTED, rotation=90, va="center")

    ax.set_xlabel("Concurrent sequences,  N")
    ax.set_ylabel("GPU memory in use  (GiB, NVML)")
    ax.set_title("Memory climbs with concurrency until the allocator has no room left", loc="left")
    ax.legend(loc="lower right", title="Experiment A", title_fontsize=8)
    return ps.save(fig, out, "fig04_memory_vs_concurrency")


def fig_regime_diagram(expA, out: Path):
    """KV-boundary / regime diagram: where the system lives, and what actually kills it."""
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    ax.axhspan(0, 100, xmin=0, xmax=1, color="#fafafa", zorder=0)
    # The region admission control believes is the danger zone.
    ax.axhspan(85, 100, color=ps.STATUS_WARN, alpha=0.13, zorder=1)
    ax.text(6, 92, "region vLLM's KV-based admission control guards against\n"
                   "(preemption / recompute)", fontsize=7.6, color="#8a6d00", linespacing=1.5)

    # The region where it actually dies.
    ax.axvspan(230, 300, color=ps.STATUS_BAD, alpha=0.10, zorder=1)
    ax.text(233, 66, "observed failure band\nN → max_num_seqs", fontsize=7.6,
            color=ps.STATUS_BAD, linespacing=1.5)

    for i, (trial, N, kv, mem, t) in enumerate(_expA_windows(expA)):
        ok = np.isfinite(N) & np.isfinite(kv)
        ax.plot(N[ok], kv[ok] * 100, color=ps.SLOTS[i], lw=1.4, alpha=0.85,
                label=f"trial {trial}", zorder=3)
        if ok.any():
            ax.scatter([N[ok][-1]], [kv[ok][-1] * 100], s=110, marker="X",
                       color=ps.STATUS_BAD, edgecolor="white", linewidth=1.2, zorder=6)

    # The safe-operating line the model would predict if KV were the constraint.
    xs = np.linspace(0, 300, 200)
    ax.plot(xs, np.minimum(xs * 0.38, 100), color=ps.INK, lw=1.3, ls=(0, (5, 2)), zorder=4,
            label="KV growth if KV were binding")

    ax.set_xlim(0, 300); ax.set_ylim(0, 105)
    ax.set_xlabel("Concurrent sequences,  N")
    ax.set_ylabel("KV cache occupancy  (%)")
    ax.set_title("Regime diagram: the failure boundary is vertical, not horizontal", loc="left")
    ax.legend(loc="upper left")
    ax.text(0.985, 0.035,
            "A KV-bound system fails on the horizontal axis (occupancy → 100%).\n"
            "This system fails on the vertical axis (N → max_num_seqs) at 28% KV.",
            transform=ax.transAxes, fontsize=7.6, color=ps.INK_2, ha="right", linespacing=1.5)
    return ps.save(fig, out, "fig09_regime_diagram")


# ======================================================================================
# fig 5 — resource bottleneck decomposition
# ======================================================================================

def fig_bottleneck(out: Path):
    """What admission control models vs what the device actually holds at failure."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 5.0),
                                   gridspec_kw={"width_ratios": [1, 1.9], "wspace": 0.06})
    for a in (axL, axR):
        a.set_axisbelow(True)          # grid behind the bars, not through them
        a.grid(axis="x", visible=False)

    # --- left: the reservation, from vLLM's own startup profile ---
    comps = [("Model weights", 2.89, ps.SLOTS[6]),
             ("Activation headroom", 2.02, ps.SLOTS[1]),
             ("KV cache", 8.15, ps.SLOTS[0]),
             ("non-torch", 0.05, ps.MUTED)]
    bottom = 0.0
    for name, v, c in comps:
        axL.bar([0], [v], bottom=bottom, width=0.86, color=c, edgecolor="white",
                linewidth=2, zorder=3)
        if v > 0.4:
            axL.text(0, bottom + v / 2, f"{name}\n{v:.2f} GiB", ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold", linespacing=1.4, zorder=4)
        bottom += v
    axL.axhline(13.11, color=ps.MUTED, ls=":", lw=1.0)
    axL.set_xlim(-0.6, 0.6); axL.set_ylim(0, 16.4)
    axL.set_xticks([]); axL.set_ylabel("GiB")
    axL.set_title("What vLLM reserves\nat startup", loc="left", fontsize=9.6, linespacing=1.5)
    axL.text(0, 13.3, "budget 13.11 GiB", ha="center", fontsize=7.4, color=ps.MUTED)

    # --- right: the state at failure. Labels sit OUTSIDE the bar with leader lines, so a
    # thin segment is never captioned by text wider than itself.
    kv_used = 8.15 * 0.284
    kv_idle = 8.15 - kv_used
    parts = [("Model weights", f"{2.89:.2f} GiB", 2.89, ps.SLOTS[6]),
             ("KV actually used", f"2.31 GiB  ({28.4:.1f}% of reserve)", kv_used, ps.SLOTS[0]),
             ("KV reserved but IDLE", "5.84 GiB never touched", kv_idle, "#cde2fb"),
             ("Activations + spec-decode scorer", f"{12.90 - 2.89 - 8.15:.2f} GiB", 12.90 - 2.89 - 8.15, ps.SLOTS[1])]
    bottom = 0.0
    for name, val, v, c in parts:
        axR.bar([0], [v], bottom=bottom, width=0.5, color=c, edgecolor="white",
                linewidth=2, zorder=3)
        ymid = bottom + v / 2
        axR.annotate(f"{name}\n{val}", xy=(0.26, ymid), xytext=(0.46, ymid),
                     fontsize=8, color=ps.INK, va="center", ha="left", linespacing=1.45,
                     arrowprops=dict(arrowstyle="-", color=ps.MUTED, lw=0.7,
                                     shrinkA=0, shrinkB=2))
        bottom += v

    # The allocation that failed.
    axR.bar([0], [0.742], bottom=13.11, width=0.5, color="none", edgecolor=ps.STATUS_BAD,
            linewidth=1.8, linestyle="--", zorder=4)
    axR.annotate("742 MiB requested by _contract_batch\n"
                 "only 736 MiB free  →  OOM",
                 xy=(0.26, 13.48), xytext=(0.46, 15.55), fontsize=8, color=ps.STATUS_BAD,
                 va="center", linespacing=1.5, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=ps.STATUS_BAD, lw=0.9,
                                 shrinkA=0, shrinkB=2))
    axR.axhline(13.11, color=ps.MUTED, ls=":", lw=1.0)
    axR.axhline(14.56, color=ps.STATUS_BAD, ls="--", lw=1.1)
    axR.text(-0.52, 14.28, "device capacity 14.56 GiB", fontsize=7.4, color=ps.STATUS_BAD)
    axR.set_xlim(-0.6, 2.55); axR.set_ylim(0, 16.4)
    axR.set_xticks([]); axR.set_yticks([])
    axR.spines["left"].set_visible(False)
    axR.set_title("What the device actually holds\nat the moment of failure", loc="left",
                  fontsize=9.6, linespacing=1.5)

    fig.suptitle("The binding resource is activation memory, not KV cache",
                 x=0.008, ha="left", fontsize=10.5, fontweight="bold", y=1.005)
    return ps.save(fig, out, "fig05_bottleneck")


# ======================================================================================
# fig 6 — aggregate SLO with CIs
# ======================================================================================

def fig_aggregate_slo(stats, out: Path):
    if not stats:
        print("  fig06 skipped — no stats"); return []
    groups = sorted({(s["config"], s["rate"]) for s in stats})
    fig, axes = plt.subplots(1, len(groups), figsize=(2.55 * len(groups) + 1.0, 4.3), sharey=True)
    if len(groups) == 1:
        axes = [axes]

    for ax, (cfg, rate) in zip(axes, groups):
        cells = {s["policy"]: s for s in stats if s["config"] == cfg and s["rate"] == rate}
        pols = [p for p in ps.POLICY_ORDER if p in cells]
        xs = np.arange(len(pols))
        means = [cells[p]["metrics"]["slo_attainment"]["mean"] for p in pols]
        errs = [cells[p]["metrics"]["slo_attainment"]["ci95_halfwidth"] for p in pols]
        # Clip the drawn interval to [0,1]: a t-interval on a proportion can leave the
        # simplex, and drawing it outside would misrepresent the attainable range.
        lo = [min(m, max(0.0, m - e)) for m, e in zip(means, errs)]
        hi = [max(m, min(1.0, m + e)) for m, e in zip(means, errs)]
        yerr = np.array([[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]])

        ax.bar(xs, means, width=0.62, color=[ps.POLICY_COLOR[p] for p in pols],
               edgecolor="white", linewidth=1.6, zorder=2)
        ax.errorbar(xs, means, yerr=yerr, fmt="none", ecolor=ps.INK_2,
                    elinewidth=1.1, capsize=3.5, zorder=3)
        for x, m in zip(xs, means):
            ax.text(x, min(1.02, m + 0.035), f"{m:.2f}", ha="center", fontsize=7.6, color=ps.INK)
        ax.set_xticks(xs); ax.set_xticklabels(pols, rotation=38, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(f"{cfg}\nλ = {rate:g}", fontsize=8.8, loc="left", linespacing=1.5)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("SLO attainment  (fraction of requests)")
    fig.suptitle("Aggregate SLO attainment by policy — error bars are 95% t-intervals on n=3",
                 x=0.008, ha="left", fontsize=10.5, fontweight="bold", y=1.01)
    fig.text(0.008, -0.06,
             "Intervals are wide because n=3 (t = 4.303). Only one contrast in the whole family "
             "survives Holm correction, and it is degenerate\n(nocap has no queue by "
             "construction). Differences below should be read as observed, not established.",
             fontsize=7.6, color=ps.MUTED, ha="left", linespacing=1.5)
    return ps.save(fig, out, "fig06_aggregate_slo")


# ======================================================================================
# fig 7 — per-class SLO heatmap
# ======================================================================================

def fig_perclass_heatmap(stats, out: Path):
    if not stats:
        print("  fig07 skipped — no stats"); return []
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", ps.SEQ_BLUE)

    groups = sorted({(s["config"], s["rate"]) for s in stats})
    fig, axes = plt.subplots(1, len(groups), figsize=(3.0 * len(groups) + 1.2, 4.0))
    if len(groups) == 1:
        axes = [axes]

    for gi, (ax, (cfg, rate)) in enumerate(zip(axes, groups)):
        cells = {s["policy"]: s for s in stats if s["config"] == cfg and s["rate"] == rate}
        pols = [p for p in ps.POLICY_ORDER if p in cells]
        classes = sorted({c for p in pols for c in cells[p]["by_class_slo"]})
        M = np.full((len(pols), len(classes)), np.nan)
        for i, p in enumerate(pols):
            for j, c in enumerate(classes):
                d = cells[p]["by_class_slo"].get(c)
                if d and d.get("n"):
                    M[i, j] = d["mean"]

        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=38, ha="right", fontsize=8)
        ax.set_yticks(range(len(pols)))
        ax.set_yticklabels(pols if gi == 0 else [""] * len(pols), fontsize=8)
        ax.set_title(f"{cfg}\nλ = {rate:g}", fontsize=8.8, loc="left", linespacing=1.5)
        ax.grid(False)
        for i in range(len(pols)):
            for j in range(len(classes)):
                if np.isfinite(M[i, j]):
                    # Value printed in every cell: the heatmap is the pattern, the number is
                    # the datum, and colour alone must not carry it.
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.4,
                            color="white" if M[i, j] > 0.55 else ps.INK, fontweight="bold")
        ax.set_xticks(np.arange(-.5, len(classes), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(pols), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.6)
        ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label("SLO attainment", fontsize=8.5, color=ps.INK_2)
    cb.outline.set_visible(False)
    fig.suptitle("Per-class SLO attainment: which tenant pays for the aggregate number",
                 x=0.008, ha="left", fontsize=10.5, fontweight="bold", y=1.02)
    return ps.save(fig, out, "fig07_perclass_slo_heatmap")


# ======================================================================================
# fig 8 — model vs measured
# ======================================================================================

def fig_model_vs_measured(model_fit, out: Path):
    if not model_fit:
        print("  fig08 skipped — no model fit"); return []
    ref = model_fit.get("per_workload", {}).get("phase1_mixed")
    if not ref or "r0_tok_s_per_seq" not in ref:
        print("  fig08 skipped — reference fit unavailable"); return []

    pts = [p for p in model_fit.get("points", [])
           if p.get("config") == "phase1_mixed" and p.get("healthy")]
    N = np.array([p["N"] for p in pts]); X = np.array([p["X"] for p in pts])
    r0, mu = ref["r0_tok_s_per_seq"], ref["mu_max_tok_s"]
    pred = np.minimum(N * r0, mu)

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.10})

    xs = np.linspace(0, max(N.max() * 1.12, 70), 400)
    ax.fill_between(xs, np.minimum(xs * ref["r0_ci95"][0], ref["mu_max_ci95"][0]),
                    np.minimum(xs * ref["r0_ci95"][1], ref["mu_max_ci95"][1]),
                    color=ps.SLOTS[0], alpha=0.14, linewidth=0, label="95% bootstrap band")
    ax.plot(xs, np.minimum(xs * r0, mu), color=ps.SLOTS[0], lw=2.0,
            label=f"μ(N) = min(N·{r0:.2f}, {mu:.1f})")
    ax.scatter(N, X, s=42, color=ps.INK, marker="o", zorder=4, edgecolor="white",
               linewidth=0.9, label="measured runs")
    ax.axvline(ref["N_star"], color=ps.MUTED, lw=0.9, ls=":")
    ps.annotate(ax, f"N* = {ref['N_star']:.1f}\n[{ref['N_star_ci95'][0]:.1f}, {ref['N_star_ci95'][1]:.1f}]",
                (ref["N_star"], mu * 0.42), (ref["N_star"] + 6, mu * 0.30))
    ax.set_ylabel("Throughput  (tok/s)")
    ax.set_title(f"Fitted service law vs measurement  —  workload phase1_mixed  "
                 f"(R² = {ref['r_squared']:.3f}, MAPE = {ref['mean_abs_pct_err']:.1f}%)", loc="left")
    ax.legend(loc="lower right")

    axr.axhline(0, color=ps.MUTED, lw=0.9)
    axr.scatter(N, X - pred, s=34, color=ps.SLOTS[1], edgecolor="white", linewidth=0.8, zorder=3)
    axr.axvline(ref["N_star"], color=ps.MUTED, lw=0.9, ls=":")
    axr.set_xlabel("Concurrent sequences,  N")
    axr.set_ylabel("Residual\n(tok/s)", fontsize=8.6)
    lim = np.abs(X - pred).max() * 1.35
    axr.set_ylim(-lim, lim)
    return ps.save(fig, out, "fig08_model_vs_measured")


# ======================================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/figures/paper")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    out = Path(args.out)

    runs = load_runs(rd)
    stats = load_json(rd / "stats" / "policy_stats.json")
    model_fit = load_json(rd / "model" / "queueing_model_fit.json")
    expA = load_json(rd / "expA" / "expA_summary_corrected.json")

    print(f"runs={len(runs)}  stats={'y' if stats else 'n'}  "
          f"model={'y' if model_fit else 'n'}  expA={'y' if expA else 'n'}")

    made = []
    made += fig_architecture(out)
    made += fig_throughput_vs_concurrency(runs, model_fit, out)
    made += fig_kv_vs_concurrency(expA, runs, out)
    made += fig_memory_vs_concurrency(expA, out)
    made += fig_bottleneck(out)
    made += fig_aggregate_slo(stats, out)
    made += fig_perclass_heatmap(stats, out)
    made += fig_model_vs_measured(model_fit, out)
    made += fig_regime_diagram(expA, out)

    print(f"\nwrote {len(made)} files to {out}:")
    for p in sorted({p.stem for p in made}):
        print(f"  {p}")


if __name__ == "__main__":
    main()
