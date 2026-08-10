"""Statistical analysis of the Phase-2 policy comparison.

Turns the per-run JSON files into a table that reports uncertainty instead of bare point
estimates: mean, SD, t-based 95% CI, Welch contrasts against a baseline policy, Hedges' g, and
Holm-adjusted p-values across the comparison family.

    python tools/analyze_stats.py --results-dir results --baseline fcfs

Outputs:
    results/stats/policy_stats.json      every cell, machine-readable
    results/stats/policy_contrasts.json  every treatment-vs-baseline contrast
    results/stats/STATISTICS.md          the report to read
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.stats import contrast, describe, holm, wilson  # noqa: E402

# Metric -> (path into the run dict, display name, unit, whether lower is better)
METRICS = {
    "throughput_tok_s": (("throughput_tok_s",), "Throughput", "tok/s", False),
    "total_latency_ms_p50": (("total_latency_ms", "p50"), "Total latency p50", "ms", True),
    "total_latency_ms_p99": (("total_latency_ms", "p99"), "Total latency p99", "ms", True),
    "ttft_ms_p50": (("ttft_ms", "p50"), "TTFT p50", "ms", True),
    "ttft_ms_p99": (("ttft_ms", "p99"), "TTFT p99", "ms", True),
    "tpot_ms_p50": (("tpot_ms", "p50"), "TPOT p50", "ms", True),
    "tpot_ms_p99": (("tpot_ms", "p99"), "TPOT p99", "ms", True),
    "queue_wait_ms_p99": (("queue_wait_ms", "p99"), "Queue wait p99", "ms", True),
    "slo_attainment": (("slo_attainment",), "SLO attainment", "frac", False),
}


def dig(d: dict, path: tuple):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def load_runs(results_dir: Path) -> list[dict]:
    """Load every per-run JSON that carries a policy label (i.e. the Phase-2 campaign)."""
    runs = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and d.get("policy"):
            d["_file"] = p.name
            runs.append(d)
    return runs


def build_cells(runs: list[dict]) -> dict:
    """Group runs into (config, rate, policy) cells."""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        cells[(r["config"], float(r["rate"]), r["policy"])].append(r)
    return cells


def cell_stats(cells: dict) -> list[dict]:
    out = []
    for (config, rate, policy), rs in sorted(cells.items()):
        rec = {
            "config": config,
            "rate": rate,
            "policy": policy,
            "n_runs": len(rs),
            "files": [r["_file"] for r in rs],
            "metrics": {},
        }
        for key, (path, _, _, _) in METRICS.items():
            vals = [dig(r, path) for r in rs]
            vals = [v for v in vals if isinstance(v, (int, float))]
            rec["metrics"][key] = describe(vals).as_dict()

        # Pooled Wilson interval for SLO: treats every measured request as a Bernoulli trial
        # rather than averaging three run-level proportions.
        succ = total = 0
        for r in rs:
            n = r.get("n_requests_measured") or 0
            s = r.get("slo_attainment")
            if isinstance(s, (int, float)) and n:
                succ += round(s * n)
                total += n
        p, lo, hi = wilson(succ, total)
        rec["slo_pooled"] = {"successes": succ, "trials": total, "p": p, "ci95_lo": lo, "ci95_hi": hi}

        # Per-class SLO across repeats.
        by_class: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            for cls, cd in (r.get("by_class") or {}).items():
                v = cd.get("slo_attainment")
                if isinstance(v, (int, float)):
                    by_class[cls].append(v)
        rec["by_class_slo"] = {c: describe(v).as_dict() for c, v in sorted(by_class.items())}

        # Per-class p99 total latency, the other half of the per-class story.
        by_class_p99: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            for cls, cd in (r.get("by_class") or {}).items():
                v = dig(cd, ("total_latency_ms", "p99"))
                if isinstance(v, (int, float)):
                    by_class_p99[cls].append(v)
        rec["by_class_totlat_p99"] = {c: describe(v).as_dict() for c, v in sorted(by_class_p99.items())}
        out.append(rec)
    return out


def build_contrasts(cells: dict, baseline: str) -> list[dict]:
    """Every non-baseline policy vs the baseline, within each (config, rate)."""
    groups: dict[tuple, dict[str, list[dict]]] = defaultdict(dict)
    for (config, rate, policy), rs in cells.items():
        groups[(config, rate)][policy] = rs

    contrasts: list[dict] = []
    for (config, rate), policies in sorted(groups.items()):
        base_runs = policies.get(baseline)
        if not base_runs:
            continue
        for policy, rs in sorted(policies.items()):
            if policy == baseline:
                continue
            for key, (path, label, unit, lower_better) in METRICS.items():
                tv = [dig(r, path) for r in rs]
                bv = [dig(r, path) for r in base_runs]
                tv = [v for v in tv if isinstance(v, (int, float))]
                bv = [v for v in bv if isinstance(v, (int, float))]
                c = contrast(key, policy, tv, baseline, bv).as_dict()
                c.update(config=config, rate=rate, label=label, unit=unit,
                         lower_is_better=lower_better)
                # "Improvement" needs a direction: lower latency is good, higher throughput is.
                d = c["diff"]
                c["improves"] = (d < 0) if lower_better else (d > 0)
                contrasts.append(c)

    # Holm across the whole family reported in this document.
    adj = holm([c["p_value"] for c in contrasts])
    for c, a in zip(contrasts, adj):
        c["p_holm"] = a
        c["significant_05_holm"] = bool(a == a and a < 0.05)
    return contrasts


def fmt_ci(m: dict, prec: int = 1) -> str:
    if not m or m.get("n", 0) == 0:
        return "—"
    if m["n"] < 2:
        return f"{m['mean']:.{prec}f} (n=1)"
    return f"{m['mean']:.{prec}f} ± {m['ci95_halfwidth']:.{prec}f}"


def write_report(stats: list[dict], contrasts: list[dict], baseline: str, out: Path) -> None:
    L: list[str] = []
    A = L.append

    A("# Statistical analysis — Phase-2 policy comparison\n")
    A("All intervals are **two-sided 95% t-intervals** on n=3 repeats per cell "
      "(multiplier t(0.975, df=2) = **4.303**, not 1.96). Effect sizes are **Hedges' g** "
      "(Cohen's d with the small-sample correction J = 0.8 at n=3). Contrasts use **Welch's "
      "t-test**; p-values are additionally reported Holm-adjusted across the whole family of "
      f"{len(contrasts)} comparisons.\n")

    A("> **Read the intervals before the means.** With n=3 the CI half-width is 2.48 SD/√n. "
      "Several cells below have intervals wide enough that the point estimate alone would be "
      "misleading; those are flagged rather than quietly reported.\n")

    configs = sorted({s["config"] for s in stats})
    for config in configs:
        A(f"\n## Workload `{config}`\n")
        rates = sorted({s["rate"] for s in stats if s["config"] == config})
        for rate in rates:
            cells = [s for s in stats if s["config"] == config and s["rate"] == rate]
            if not cells:
                continue
            A(f"\n### λ = {rate:g} req/s\n")
            A("| Policy | n | Throughput (tok/s) | Total lat p50 (ms) | Total lat p99 (ms) | "
              "Queue wait p99 (ms) | SLO (repeat mean) | SLO (pooled Wilson) |")
            A("|---|---|---|---|---|---|---|---|")
            for c in sorted(cells, key=lambda x: x["policy"]):
                m = c["metrics"]
                sp = c["slo_pooled"]
                pooled = (f"{sp['p']:.3f} [{sp['ci95_lo']:.3f}, {sp['ci95_hi']:.3f}]"
                          if sp["trials"] else "—")
                A(f"| `{c['policy']}`{' *(baseline)*' if c['policy'] == baseline else ''} "
                  f"| {c['n_runs']} "
                  f"| {fmt_ci(m['throughput_tok_s'])} "
                  f"| {fmt_ci(m['total_latency_ms_p50'])} "
                  f"| {fmt_ci(m['total_latency_ms_p99'])} "
                  f"| {fmt_ci(m['queue_wait_ms_p99'])} "
                  f"| {fmt_ci(m['slo_attainment'], 3)} "
                  f"| {pooled} |")

            # Cells whose noise swamps the estimate.
            noisy = [
                (c["policy"], k, c["metrics"][k]["cv_pct"])
                for c in cells for k in ("throughput_tok_s", "total_latency_ms_p99")
                if c["metrics"][k].get("cv_pct", 0) and c["metrics"][k]["cv_pct"] > 30
            ]
            if noisy:
                A("\n**High-variance cells (CV > 30%)** — treat these means as indicative only:\n")
                for pol, k, cv in noisy:
                    A(f"- `{pol}` / {k}: CV = {cv:.0f}%")

    # --- per-class SLO ---
    A("\n## Per-class SLO attainment\n")
    A("Mean ± 95% CI over 3 repeats, by workload class.\n")
    for config in configs:
        rates = sorted({s["rate"] for s in stats if s["config"] == config})
        for rate in rates:
            cells = [s for s in stats if s["config"] == config and s["rate"] == rate]
            classes = sorted({c for s in cells for c in s["by_class_slo"]})
            if not classes:
                continue
            A(f"\n**`{config}`, λ = {rate:g}**\n")
            A("| Policy | " + " | ".join(classes) + " |")
            A("|---" * (len(classes) + 1) + "|")
            for c in sorted(cells, key=lambda x: x["policy"]):
                row = [fmt_ci(c["by_class_slo"].get(cl, {}), 3) for cl in classes]
                A(f"| `{c['policy']}` | " + " | ".join(row) + " |")

    # --- contrasts ---
    A(f"\n## Contrasts vs `{baseline}`\n")
    A("`Δ` is treatment − baseline with its own 95% CI. `g` is Hedges' g "
      "(|g|<0.2 negligible, <0.5 small, <0.8 medium, ≥0.8 large). "
      "`p` is Welch; `p_holm` is Holm-adjusted across all comparisons in this document.\n")

    shown = {"throughput_tok_s", "total_latency_ms_p50", "total_latency_ms_p99", "slo_attainment"}
    for config in configs:
        rates = sorted({c["rate"] for c in contrasts if c["config"] == config})
        for rate in rates:
            sub = [c for c in contrasts
                   if c["config"] == config and c["rate"] == rate and c["metric"] in shown]
            if not sub:
                continue
            A(f"\n**`{config}`, λ = {rate:g}**\n")
            A("| Policy | Metric | baseline | treatment | Δ [95% CI] | g | magnitude | p | p_holm |")
            A("|---|---|---|---|---|---|---|---|---|")
            for c in sorted(sub, key=lambda x: (x["treatment"], x["metric"])):
                star = " ✓" if c.get("significant_05_holm") else ""
                A(f"| `{c['treatment']}` | {c['label']} "
                  f"| {c['mean_base']:.3g} | {c['mean_treat']:.3g} "
                  f"| {c['diff']:+.3g} [{c['diff_ci95_lo']:+.3g}, {c['diff_ci95_hi']:+.3g}] "
                  f"| {c['hedges_g']:+.2f} | {c['magnitude']} "
                  f"| {c['p_value']:.3f} | {c['p_holm']:.3f}{star} |")

    # --- honest summary ---
    sig = [c for c in contrasts if c.get("significant_05_holm")]
    raw_sig = [c for c in contrasts if c.get("significant_05")]
    A("\n## What survives\n")
    A(f"- Comparisons made: **{len(contrasts)}**")
    A(f"- Significant at p < 0.05 **before** correction: **{len(raw_sig)}**")
    A(f"- Significant **after** Holm correction: **{len(sig)}**")
    if sig:
        A("\nSurviving contrasts:\n")
        for c in sorted(sig, key=lambda x: x["p_holm"]):
            A(f"- `{c['treatment']}` vs `{c['baseline']}` — {c['label']} "
              f"({c['config']}, λ={c['rate']:g}): {c['diff']:+.3g} {c['unit']}, "
              f"g = {c['hedges_g']:+.2f}, p_holm = {c['p_holm']:.4f}")
    else:
        A("\n**No contrast survives Holm correction at n=3.** This is a statement about "
          "statistical power, not about the absence of an effect: with 3 repeats per cell, "
          "Welch's t-test can only detect very large effects. Large point differences with "
          "wide intervals are best reported as *observed differences pending replication*, "
          "and the honest fix is more repeats, not a weaker correction.")

    A("\n## Caveats\n")
    A("1. **n=3 per cell.** Every interval is wide and every effect size is imprecise. "
      "Raising repeats to 10 would shrink the t-multiplier from 4.303 to 2.262 and the "
      "half-width by roughly 3x.")
    A("2. **Repeats are not independent of drift.** All repeats of a cell ran back to back on "
      "one passively cooled T4, so a thermal excursion is shared within a cell rather than "
      "averaged out. Check `thermal_throttled` before trusting a narrow interval.")
    A("3. **SLO is a proportion.** The repeat-mean column and the pooled Wilson column answer "
      "different questions; where they disagree, the pooled interval is the better guide to "
      "per-request behaviour and the repeat interval to run-to-run stability.")
    A("4. **Holm assumes the family is the one reported here.** Adding metrics later without "
      "re-running the correction reintroduces the multiplicity it removes.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--baseline", default="fcfs")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(results_dir)
    if not runs:
        sys.exit(f"no policy-labelled runs found in {results_dir}")
    cells = build_cells(runs)
    stats = cell_stats(cells)
    contrasts = build_contrasts(cells, args.baseline)

    (out_dir / "policy_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (out_dir / "policy_contrasts.json").write_text(json.dumps(contrasts, indent=2) + "\n")
    write_report(stats, contrasts, args.baseline, out_dir / "STATISTICS.md")

    print(f"runs loaded      : {len(runs)}")
    print(f"cells            : {len(stats)}  ({len({(s['config'], s['rate']) for s in stats})} config x rate groups)")
    print(f"contrasts        : {len(contrasts)} vs '{args.baseline}'")
    print(f"significant raw  : {sum(1 for c in contrasts if c.get('significant_05'))}")
    print(f"significant Holm : {sum(1 for c in contrasts if c.get('significant_05_holm'))}")
    print(f"wrote            : {out_dir}/STATISTICS.md, policy_stats.json, policy_contrasts.json")


if __name__ == "__main__":
    main()
