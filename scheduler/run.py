"""Phase-2 runner: policy comparison matrix + the SRPT length-prediction mini-experiment.

Admission/ordering layer in FRONT of vLLM (never touches engine internals). Reuses the Phase-1
harness (workload, telemetry, aggregation, plots).

On the serving box (venv active), server already up with n-gram spec decode:
    # policy comparison across policies x rates x repeats
    python scheduler/run.py --config configs/phase2_mixed.yaml
    # length-prediction mini-experiment (SRPT oracle vs prediction vs prompt-proxy)
    python scheduler/run.py --config configs/phase2_mixed.yaml --experiment predictor
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config  # noqa: E402
from common.metrics import aggregate_run, summarize_repeats, write_results  # noqa: E402
from common.plots import make_policy_figures, make_class_figures  # noqa: E402
from common.telemetry import TelemetrySampler, wait_for_cool  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scheduler import make_policy, run_scheduled_load, prompt_proxy_size  # noqa: E402
from predictor import (  # noqa: E402
    LengthPredictor, actuals_by_seq, oracle_size_fn, prediction_size_fn,
)


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


def _sampler(cfg):
    tel = cfg.get("telemetry", {}) or {}
    return TelemetrySampler(
        metrics_url=f"{cfg['server']['base_url']}/metrics",
        interval_s=tel.get("interval_s", 0.5),
        gpu_index=tel.get("gpu_index", 0),
    )


async def _one_run(cfg, rate, seed, policy, size_fn, results_dir, tag):
    # Thermal gate: block until the (passively-cooled) T4 is below threshold so all runs measure at
    # full clocks and are comparable. Runs that still throttle are flagged in the results JSON.
    tel = cfg.get("telemetry", {}) or {}
    if tel.get("cooldown_c") is not None:
        await asyncio.to_thread(wait_for_cool, float(tel["cooldown_c"]), tel.get("gpu_index", 0))
    sampler = _sampler(cfg)
    sampler.start()
    results, t0 = await run_scheduled_load(cfg, rate, seed, policy, sampler, size_fn=size_fn)
    sampler.stop()
    agg = aggregate_run(cfg, results, sampler, t0, rate, seed, repeat=int(tag.split("rep")[-1]) if "rep" in tag else 0)
    agg["policy"] = policy.name
    agg["cap"] = cfg.get("scheduler", {}).get("max_in_flight")
    agg["label"] = tag
    return agg, results


async def run_matrix(cfg, policies, results_dir: Path):
    base_url = cfg["server"]["base_url"]
    wl = cfg["workload"]
    rates = [float(r) for r in wl["arrival_rates"]]
    repeats = int(wl.get("repeats", 3))
    base_seed = int(wl.get("seed", 12345))

    await wait_for_health(base_url)
    print(f"[p2] policies={policies} rates={rates} x {repeats} repeats "
          f"(cap={cfg.get('scheduler', {}).get('max_in_flight')}).")

    all_runs = []
    for pol_name in policies:
        for rate in rates:
            for rep in range(repeats):
                policy = make_policy(pol_name)  # fresh state per run
                agg, _ = await _one_run(cfg, rate, base_seed + rep, policy, prompt_proxy_size,
                                        results_dir, f"{pol_name}_r{rate:g}_rep{rep}")
                write_results(agg, results_dir)
                all_runs.append(agg)
                print(f"[p2] {pol_name} λ={rate:g} rep={rep}: "
                      f"SLO={agg['slo_attainment']:.2f} "
                      f"totlat_p99={agg['total_latency_ms']['p99']:.0f}ms "
                      f"qwait_p99={agg['queue_wait_ms']['p99']:.0f}ms "
                      f"TPOT_p50={agg['tpot_ms']['p50']:.0f}ms n={agg['n_requests_measured']}")

    figs = make_policy_figures(all_runs, results_dir / "figures", prefix=cfg.get("_config_name"))
    figs += make_class_figures(all_runs, results_dir / "figures", prefix=cfg.get("_config_name"))
    summ = {pol: [summarize_repeats([r for r in all_runs if r["policy"] == pol and r["rate"] == rate])
                  for rate in rates] for pol in policies}
    summ_path = results_dir / "summaries" / f"{cfg.get('_config_name')}_policy_summary.json"
    summ_path.parent.mkdir(parents=True, exist_ok=True)
    summ_path.write_text(json.dumps(summ, indent=2))
    print(f"[p2] wrote {summ_path.name} and {len(figs)} figures:")
    for f in figs:
        print(f"     {f}")


async def run_predictor_experiment(cfg, results_dir: Path):
    """SRPT ordering under three size sources: oracle vs prediction vs prompt-proxy."""
    base_url = cfg["server"]["base_url"]
    wl = cfg["workload"]
    rate = float(wl["arrival_rates"][len(wl["arrival_rates"]) // 2])  # a mid-load rate
    seed = int(wl.get("seed", 12345))
    await wait_for_health(base_url)
    print(f"[p2-pred] profiling run at λ={rate:g} (FCFS) to learn actual lengths + train predictor ...")

    # 1) Profiling run (FCFS) -> actual lengths + predictor training data.
    prof_agg, prof_results = await _one_run(cfg, rate, seed, make_policy("fcfs"), prompt_proxy_size,
                                            results_dir, "profile")
    actuals = actuals_by_seq(prof_results)
    predictor = LengthPredictor().fit(prof_results)
    mae = predictor.mae(prof_results)
    print(f"[p2-pred] predictor MAE = {mae:.1f} tokens over {len(actuals)} requests")

    # 2) Three SRPT variants on the same seed/rate.
    variants = {
        "srpt-oracle": oracle_size_fn(actuals),
        "srpt-prediction": prediction_size_fn(predictor),
        "srpt-prompt-proxy": prompt_proxy_size,
    }
    rows = []
    for name, size_fn in variants.items():
        agg, _ = await _one_run(cfg, rate, seed, make_policy("srpt"), size_fn, results_dir, name)
        agg["policy"] = name
        agg["predictor_mae_tokens"] = mae
        write_results(agg, results_dir)
        rows.append(agg)
        print(f"[p2-pred] {name}: SLO={agg['slo_attainment']:.2f} "
              f"totlat_p99={agg['total_latency_ms']['p99']:.0f}ms "
              f"qwait_p50={agg['queue_wait_ms']['p50']:.0f}ms")

    out = results_dir / "summaries" / f"{cfg.get('_config_name')}_predictor_experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rate": rate, "seed": seed, "predictor_mae_tokens": mae,
        "variants": {r["policy"]: {
            "slo_attainment": r["slo_attainment"],
            "total_latency_ms_p99": r["total_latency_ms"]["p99"],
            "queue_wait_ms_p50": r["queue_wait_ms"]["p50"],
        } for r in rows},
    }, indent=2))
    print(f"[p2-pred] wrote {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--policies", default="fcfs,srpt,edf,adaptive")
    ap.add_argument("--experiment", choices=["matrix", "predictor"], default="matrix")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rd = Path(args.results_dir)
    if args.experiment == "predictor":
        asyncio.run(run_predictor_experiment(cfg, rd))
    else:
        asyncio.run(run_matrix(cfg, [p.strip() for p in args.policies.split(",")], rd))


if __name__ == "__main__":
    main()
