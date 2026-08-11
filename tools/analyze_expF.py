"""Experiment F analysis — does the scorer memory law hold on its (k+1) term?

The law is

    M_scorer = N * (k+1) * V * 4 bytes

and until now only N had ever been varied: every observed death used k=4, on two models that
share V=151936. The coefficient was fitted on one axis and asserted on three.

F varies k at the default cap, where the scorer boundary, the sequence cap and the KV ceiling
land at three well-separated concurrencies instead of on top of each other. The test statistic is
**MiB per sequence** — the failed allocation divided by concurrency at the failing step — because
that is the only quantity (k+1) controls. The concurrency at which each arm dies is *not* the
test: it depends on how much headroom happened to be free, which varies with the workload's
context lengths. Two arms dying at different N with the same MiB/seq would refute the law; two
arms dying at the same N with different MiB/seq would confirm it.

Predictions, fixed before the runs:

    k=2  ->  1.739 MiB/seq   and no death at all (boundary at N ~ 344, past the cap of 256)
    k=4  ->  2.898 MiB/seq   (already measured: 741.9 MiB at N=256)
    k=7  ->  4.637 MiB/seq

Usage:
    python tools/analyze_expF.py --results-dir results/expF
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

V_QWEN = 151936
BYTES_PER_ELEM = 4  # the scorer's probability tensor is fp32 regardless of the model's dtype

# arm tag -> (k, vocab, expects_death). The k=2 arm's prediction is a null: its boundary sits
# beyond the sequence cap, so a death there would falsify the law rather than support it.
ARMS = {
    "k2_cap256": (2, V_QWEN, False),
    "k7_cap256": (7, V_QWEN, True),
}

# Reference point from experiment B, carried in so the ratio test has three k values rather than
# two. Not re-measured here.
K4_REFERENCE = {"k": 4, "mib_per_seq": 741.9 / 256, "alloc_mib": 741.9, "n": 256}


def predicted_mib_per_seq(k: int, vocab: int) -> float:
    return (k + 1) * vocab * BYTES_PER_ELEM / 2**20


def load_arm(results_dir: Path, tag: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(results_dir / tag / "*trial*.json"))):
        d = json.loads(Path(f).read_text())
        if isinstance(d, list):  # summary files are lists; trial files are not
            continue
        oom = d.get("oom") or {}
        peaks = d.get("peaks") or {}
        at_fail = d.get("at_failure") or {}
        n = at_fail.get("num_running") or peaks.get("num_running")
        alloc = oom.get("failed_alloc_mib")
        out.append({
            "trial": d.get("trial"),
            "died": bool(d.get("died")),
            "alloc_mib": alloc,
            "n": n,
            "peak_n": peaks.get("num_running"),
            "peak_kv": peaks.get("kv_occupancy"),
            "site": oom.get("alloc_site"),
            "mib_per_seq": (alloc / n) if (alloc and n) else None,
            "file": Path(f).name,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results/expF")
    ap.add_argument("--out", default="results/expF/expF_analysis.json")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    report: dict = {"arms": {}, "reference_k4": K4_REFERENCE}

    print("=" * 78)
    print("EXPERIMENT F — the (k+1) term of the scorer memory law")
    print("=" * 78)
    print(f"{'arm':<12} {'k':>3} {'trial':>5} {'died':>5} {'alloc MiB':>10} {'N':>6} "
          f"{'MiB/seq':>9} {'predicted':>10} {'err %':>7}")

    for tag, (k, vocab, expects_death) in ARMS.items():
        pred = predicted_mib_per_seq(k, vocab)
        rows = load_arm(rd, tag)
        report["arms"][tag] = {
            "k": k, "vocab": vocab, "predicted_mib_per_seq": pred,
            "expects_death": expects_death, "trials": rows,
        }
        if not rows:
            print(f"{tag:<12} {k:>3}   (no trials found)")
            continue
        for r in rows:
            mps = r["mib_per_seq"]
            err = (100 * (mps - pred) / pred) if mps else float("nan")
            print(f"{tag:<12} {k:>3} {r['trial']:>5} {str(r['died']):>5} "
                  f"{(r['alloc_mib'] if r['alloc_mib'] else float('nan')):>10.1f} "
                  f"{(r['n'] if r['n'] else float('nan')):>6.0f} "
                  f"{(mps if mps else float('nan')):>9.3f} {pred:>10.3f} {err:>7.1f}")

        deaths = [r for r in rows if r["died"]]
        measured = [r["mib_per_seq"] for r in rows if r["mib_per_seq"]]
        arm = report["arms"][tag]
        arm["n_died"] = len(deaths)
        arm["n_trials"] = len(rows)
        if measured:
            arm["mean_mib_per_seq"] = statistics.fmean(measured)
            arm["rel_error_pct"] = 100 * (arm["mean_mib_per_seq"] - pred) / pred

        # A null arm is only evidence if it was actually pushed hard. Record how close it got.
        if not expects_death:
            arm["peak_n_observed"] = max((r["peak_n"] or 0) for r in rows)
            arm["peak_kv_observed"] = max((r["peak_kv"] or 0) for r in rows)

    print()
    print("-" * 78)
    print("VERDICT")
    print("-" * 78)

    k7 = report["arms"].get("k7_cap256", {})
    k2 = report["arms"].get("k2_cap256", {})

    # An arm with no trials is a campaign that did not finish, not a null result. Saying so
    # matters because this report is read unattended, and "k=2 survived all 0 trials, as
    # predicted" is the kind of sentence that gets believed.
    missing = [t for t in ARMS if not report["arms"].get(t, {}).get("n_trials")]
    if missing:
        print(f"INCOMPLETE: no trials found for {', '.join(missing)}.")
        print("The campaign did not finish, or the arm crashed before writing. Check expF_run.log.")
        print("Nothing below should be read as a result for a missing arm.")
        print()

    if k7.get("n_died"):
        obs = k7.get("mean_mib_per_seq")
        pred = k7["predicted_mib_per_seq"]
        err = k7.get("rel_error_pct", float("nan"))
        print(f"k=7 died in {k7['n_died']}/{k7['n_trials']} trials at {obs:.3f} MiB/seq "
              f"vs {pred:.3f} predicted ({err:+.1f}%).")
        ratio = obs / K4_REFERENCE["mib_per_seq"]
        print(f"  ratio to k=4: {ratio:.3f} observed vs {8 / 5:.3f} predicted (8/5).")
        # The concurrency argument is what breaks the confound experiment C could not.
        ns = [r["n"] for r in k7["trials"] if r["died"] and r["n"]]
        if ns:
            print(f"  died at N = {sorted(ns)}, against a sequence cap of 256 — the cap cannot")
            print("  explain a death at half its value, so the boundary is the scorer's.")
    else:
        print("k=7 did NOT die. Either the ramp never reached the boundary, or the law is wrong")
        print("about (k+1). Check peak N and KV occupancy before concluding anything: if KV")
        print("pinned near 99% the arm went KV-bound again and the test did not run.")
        if k7.get("trials"):
            print(f"  peak N observed: {max((r['peak_n'] or 0) for r in k7['trials'])}, "
                  f"peak KV: {max((r['peak_kv'] or 0) for r in k7['trials']):.3f}")

    if k2.get("n_trials"):
        if k2.get("n_died"):
            print(f"k=2 DIED in {k2['n_died']}/{k2['n_trials']} trials — this was predicted not to "
                  "happen.")
            print("  A death here is at a concurrency the scorer cannot account for, and falsifies")
            print("  the law as stated. Do not report the k=7 agreement without resolving this.")
        else:
            print(f"k=2 survived all {k2.get('n_trials', 0)} trials, as predicted "
                  f"(boundary at N ~ 344 lies past the cap of 256).")
            print(f"  pushed to peak N = {k2.get('peak_n_observed')}, "
                  f"peak KV = {k2.get('peak_kv_observed', float('nan')):.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
