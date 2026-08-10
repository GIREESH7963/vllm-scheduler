"""Recompute Experiment A's derived fields from preserved raw data.

The probe stores the full pre-failure telemetry window and the complete server log, so the
derived summary can be corrected without re-running any GPU time. Two corrections matter:

1. **Allocation site.** vLLM logs the OOM twice — once from the engine (the real stack) and once
   per outstanding request as the cached ``MQEngineDeadError`` is replayed by the API layer.
   Reading forwards from the first match lands in the replay and misattributes the failure to
   ``client.py:_process_request``. Forensics are re-derived from the engine-prefixed block.

2. **State at failure.** ``/metrics`` keeps serving the last gauge values for a moment after the
   engine dies, while NVML already reports the memory as released. Selecting "the last sample
   with a parseable num_running" therefore pairs a live concurrency reading with a post-mortem
   memory reading. Liveness is instead determined from resident GPU memory.

    python tools/reanalyze_expA.py --dir results/expA
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase3-limits"))

from oom_probe import parse_oom  # noqa: E402

# The engine's resident footprint is >10 GiB; anything below this means it has exited.
LIVE_MEM_FRAC = 0.5


def recompute_at_failure(window: list[dict]) -> dict | None:
    """Last sample taken while the engine was still resident on the GPU."""
    mems = [s.get("mem_used_gib") for s in window if isinstance(s.get("mem_used_gib"), (int, float))]
    if not mems:
        return None
    peak_mem = max(mems)
    threshold = peak_mem * LIVE_MEM_FRAC

    live = [s for s in window if isinstance(s.get("mem_used_gib"), (int, float))
            and s["mem_used_gib"] >= threshold]
    if not live:
        return None
    last = live[-1]

    def _f(key):
        vals = [s.get(key) for s in live
                if isinstance(s.get(key), (int, float)) and s.get(key) == s.get(key)]
        return vals

    running = _f("num_running")
    kv = _f("kv_occupancy")
    temps = _f("temp_c")

    return {
        "engine_live_samples": len(live),
        "peak_mem_used_gib": peak_mem,
        # Values at the final live sample.
        "num_running": last.get("num_running"),
        "num_waiting": last.get("num_waiting"),
        "kv_occupancy": last.get("kv_occupancy"),
        "mem_total_gib": last.get("mem_total_gib"),
        "mem_used_gib": last.get("mem_used_gib"),
        "mem_free_gib": last.get("mem_free_gib"),
        "util_gpu_pct": last.get("util_gpu_pct"),
        "util_mem_pct": last.get("util_mem_pct"),
        "temp_c": last.get("temp_c"),
        "power_w": last.get("power_w"),
        "thermal_throttled": last.get("thermal_throttled"),
        # Peaks over the live window — the boundary claim is about the maximum reached.
        "peak_num_running": max(running) if running else None,
        "peak_kv_occupancy": max(kv) if kv else None,
        "max_temp_c": max(temps) if temps else None,
        "any_thermal_throttle": any(bool(s.get("thermal_throttled")) for s in live),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/expA")
    args = ap.parse_args()
    d = Path(args.dir)

    files = sorted(d.glob("*_expA_trial*.json"))
    if not files:
        sys.exit(f"no trial files in {d}")

    out = []
    for f in files:
        rec = json.loads(f.read_text())
        window = rec.get("pre_failure_window") or []

        af = recompute_at_failure(window)
        if af:
            rec["at_failure_corrected"] = af

        log = rec.get("server_log")
        log_path = Path(log) if log and Path(log).exists() else d / "logs" / Path(str(log)).name
        if log_path.exists():
            oom = parse_oom(log_path)
            if oom:
                rec["oom_corrected"] = oom

        f.write_text(json.dumps(rec, indent=2, default=str) + "\n")
        out.append(rec)

    print(f"{'trial':>5} {'died':>5} {'alloc MiB':>10} {'site':>38} {'peak run':>9} "
          f"{'peak kv%':>9} {'mem GiB':>8} {'temp':>5} {'thr':>4}")
    print("-" * 104)
    for r in out:
        oom = r.get("oom_corrected") or r.get("oom") or {}
        af = r.get("at_failure_corrected") or {}
        kv = af.get("peak_kv_occupancy")
        print(f"{r['trial']:>5} {str(r.get('died')):>5} "
              f"{oom.get('failed_alloc_mib', float('nan')):>10.1f} "
              f"{str(oom.get('alloc_site'))[:38]:>38} "
              f"{af.get('peak_num_running', float('nan')):>9.0f} "
              f"{(kv * 100 if kv is not None else float('nan')):>9.2f} "
              f"{af.get('mem_used_gib', float('nan')):>8.2f} "
              f"{af.get('max_temp_c', float('nan')):>5.0f} "
              f"{str(af.get('any_thermal_throttle'))[:4]:>4}")

    summ = d / "expA_summary_corrected.json"
    summ.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"\nwrote {summ}")


if __name__ == "__main__":
    main()
