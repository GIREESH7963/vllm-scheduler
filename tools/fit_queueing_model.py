"""Fit the concurrency-throughput law mu(N) = min(N * r0, mu_max).

The model has exactly two free parameters and one derived quantity:

    r0      per-sequence service rate in the uncontended regime  [tok/s/seq]
    mu_max  saturated aggregate service rate                     [tok/s]
    N*      = mu_max / r0, the concurrency at which the system leaves the linear regime

Both parameters are **empirically fitted**, not derived from hardware specs. The fit is done per
workload, because r0 depends on the prompt/output length mix: a workload of long generations
spends proportionally more time in memory-bandwidth-bound decode than one of short ones, and a
single pooled r0 would average two different physical regimes.

Uncertainty comes from a non-parametric bootstrap over runs (not over residuals), which keeps the
run-to-run variance structure — the dominant noise source here — intact.

    python tools/fit_queueing_model.py --results-dir results
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def mu_model(N, r0, mu_max):
    """Piecewise-linear saturating service rate."""
    return np.minimum(np.asarray(N, dtype=float) * r0, mu_max)


def load_points(results_dir: Path) -> list[dict]:
    """Collect (N, X) observations from every healthy run.

    Runs whose engine died are excluded: a run that OOM'd mid-window reports a throughput
    averaged over a period when the engine was partly dead, which is not a point on the service
    curve at all. Including them drags the saturated asymptote down and would make the model look
    like it saturates far earlier than it does.
    """
    pts = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or "num_running_mean" not in d:
            continue
        N = d.get("num_running_mean")
        X = d.get("throughput_tok_s")
        nfail = d.get("n_requests_failed") or 0
        nmeas = d.get("n_requests_measured") or 0
        if not isinstance(N, (int, float)) or N != N or not isinstance(X, (int, float)):
            continue
        healthy = nmeas > 0 and nfail == 0
        pts.append({
            "file": p.name,
            "config": d.get("config"),
            "policy": d.get("policy"),
            "rate": d.get("rate"),
            "N": float(N),
            "X": float(X),
            "tpot_p50_ms": (d.get("tpot_ms") or {}).get("p50"),
            "kv_occupancy": d.get("kv_occupancy"),
            "acceptance_rate": d.get("acceptance_rate"),
            "sm_active_pct": d.get("sm_active_pct"),
            "n_failed": nfail,
            "n_measured": nmeas,
            "healthy": healthy,
        })
    return pts


def fit(points: list[dict], n_boot: int = 2000, seed: int = 12345) -> dict:
    N = np.array([p["N"] for p in points], float)
    X = np.array([p["X"] for p in points], float)
    if N.size < 4:
        return {"error": f"need >=4 points, got {N.size}", "n_points": int(N.size)}

    # Sensible starts: r0 from the lowest-N observations, mu_max from the highest throughputs.
    lo = N <= max(np.quantile(N, 0.25), 1e-9)
    r0_0 = float(np.median(X[lo] / np.maximum(N[lo], 1e-9))) if lo.any() else 50.0
    mu0 = float(np.quantile(X, 0.95))

    try:
        popt, _ = curve_fit(mu_model, N, X, p0=[max(r0_0, 1e-3), max(mu0, 1e-3)],
                            bounds=([1e-6, 1e-6], [np.inf, np.inf]), maxfev=20000)
    except Exception as e:
        return {"error": f"fit failed: {e}", "n_points": int(N.size)}

    r0, mu_max = float(popt[0]), float(popt[1])
    pred = mu_model(N, r0, mu_max)
    resid = X - pred
    ss_res = float((resid**2).sum())
    ss_tot = float(((X - X.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Bootstrap over runs.
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, N.size, N.size)
        try:
            pb, _ = curve_fit(mu_model, N[idx], X[idx], p0=[r0, mu_max],
                              bounds=([1e-6, 1e-6], [np.inf, np.inf]), maxfev=8000)
            if np.isfinite(pb).all():
                boots.append(pb)
        except Exception:
            continue
    B = np.array(boots) if boots else np.empty((0, 2))

    def ci(col):
        return ([float(np.percentile(B[:, col], 2.5)), float(np.percentile(B[:, col], 97.5))]
                if B.size else [float("nan")] * 2)

    n_star = mu_max / r0 if r0 > 0 else float("nan")
    n_star_boot = B[:, 1] / B[:, 0] if B.size else np.array([])

    # Residual structure in the saturated regime: if the plateau is not flat, min() is the
    # wrong functional form and the model should say so rather than hide it in R^2.
    sat = N >= n_star
    sat_slope = float("nan")
    if sat.sum() >= 3:
        sat_slope = float(np.polyfit(N[sat], X[sat], 1)[0])

    return {
        "n_points": int(N.size),
        "r0_tok_s_per_seq": r0,
        "r0_ci95": ci(0),
        "mu_max_tok_s": mu_max,
        "mu_max_ci95": ci(1),
        "N_star": float(n_star),
        "N_star_ci95": ([float(np.percentile(n_star_boot, 2.5)),
                         float(np.percentile(n_star_boot, 97.5))] if n_star_boot.size else [float("nan")] * 2),
        "r_squared": float(r2),
        "rmse_tok_s": float(np.sqrt(ss_res / N.size)),
        "mean_abs_pct_err": float(np.mean(np.abs(resid) / np.maximum(X, 1e-9)) * 100),
        "n_below_knee": int((N < n_star).sum()),
        "n_above_knee": int((N >= n_star).sum()),
        "saturated_slope_tok_s_per_seq": sat_slope,
        "N_range": [float(N.min()), float(N.max())],
        "bootstrap_success": int(B.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--min-points", type=int, default=6)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    pts = load_points(results_dir)
    healthy = [p for p in pts if p["healthy"]]
    print(f"observations: {len(pts)} total, {len(healthy)} healthy "
          f"({len(pts) - len(healthy)} excluded for failed requests)")

    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for p in healthy:
        by_cfg[p["config"]].append(p)

    out = {
        "pooled": fit(healthy),
        "per_workload": {},
        "points": pts,
        "notes": {
            "model": "mu(N) = min(N * r0, mu_max)",
            "excluded_unhealthy": len(pts) - len(healthy),
            "fit_method": "non-linear least squares (Trust Region Reflective), "
                          "bootstrap over runs, 2000 resamples",
        },
    }
    for cfg, ps in sorted(by_cfg.items()):
        if len(ps) >= args.min_points:
            out["per_workload"][cfg] = fit(ps)

    (out_dir / "queueing_model_fit.json").write_text(json.dumps(out, indent=2) + "\n")

    def show(name: str, f: dict) -> None:
        if "error" in f:
            print(f"\n{name}: {f['error']}")
            return
        print(f"\n{name}  (n={f['n_points']}, N in [{f['N_range'][0]:.2f}, {f['N_range'][1]:.1f}])")
        print(f"  r0      = {f['r0_tok_s_per_seq']:8.2f} tok/s/seq  "
              f"95% CI [{f['r0_ci95'][0]:.2f}, {f['r0_ci95'][1]:.2f}]")
        print(f"  mu_max  = {f['mu_max_tok_s']:8.1f} tok/s       "
              f"95% CI [{f['mu_max_ci95'][0]:.1f}, {f['mu_max_ci95'][1]:.1f}]")
        print(f"  N*      = {f['N_star']:8.2f} seqs        "
              f"95% CI [{f['N_star_ci95'][0]:.2f}, {f['N_star_ci95'][1]:.2f}]")
        print(f"  R^2     = {f['r_squared']:8.4f}   RMSE = {f['rmse_tok_s']:.1f} tok/s  "
              f"MAPE = {f['mean_abs_pct_err']:.1f}%")
        print(f"  points below/above knee: {f['n_below_knee']}/{f['n_above_knee']}   "
              f"plateau slope = {f['saturated_slope_tok_s_per_seq']:.3f} tok/s/seq")

    show("POOLED (all workloads)", out["pooled"])
    for cfg, f in out["per_workload"].items():
        show(f"workload `{cfg}`", f)

    print(f"\nwrote {out_dir / 'queueing_model_fit.json'}")


if __name__ == "__main__":
    main()
