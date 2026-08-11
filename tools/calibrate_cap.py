"""Measure how `max_num_seqs` trades activation headroom against KV cache.

Experiments C and E were both run at `max_num_seqs=1024` on the assumption that the flag is a
scheduler cap — a ceiling on concurrency and nothing more. It is not. vLLM sizes its memory
profiling pass at `max_num_seqs`, reserves the resulting activation peak, and hands KV whatever
survives. Raising the cap therefore *shrinks* the KV pool, and on a 16 GiB T4 it shrinks it
enough to change which resource binds:

    cap  256 -> activation 2.52 GiB, KV 4.74 GiB, "Maximum concurrency" 4.22x -> scorer-bound
    cap 1024 -> activation 5.64 GiB, KV 1.63 GiB, "Maximum concurrency" 1.44x -> KV-bound

Both C arms and both E arms ran with the 1.63 GiB pool, went KV-bound near N ~ 125, and so never
reached the scorer boundary they were built to measure. Neither experiment answered its question.

This tool measures the trade-off directly so the next round picks its cap from data instead of
assumption. For each cap it starts a server, waits for the profiling line, records the split, and
kills the server before any workload runs — roughly 50 s per point and no GPU load, because the
number wanted is decided during startup.

The activation peak is **independent of `num_speculative_tokens`** (5.64 GiB at k=2, 4 and 7
alike), which is why the sweep varies only the cap: vLLM's profiling pass never exercises the
speculative scorer, so the scorer's [N x (k+1) x V] tensor is absent from the reservation that is
supposed to bound memory use. That is the paper's central mechanism, visible in a startup log.
The k arms below re-measure it at cap=256 to confirm the independence holds away from 1024.

Usage:
    python tools/calibrate_cap.py --out results/expF/calibration.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# `phase3-limits` is not a legal module name, so the probe's server lifecycle is loaded by path
# rather than imported. Reusing it matters: process-group teardown and the GPU-free wait are the
# reason consecutive trials start from the same memory state, and a second copy of that logic
# here would be a second thing to get wrong.
_spec = importlib.util.spec_from_file_location("oom_probe", _ROOT / "phase3-limits" / "oom_probe.py")
_oom_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oom_probe)

VllmServer = _oom_probe.VllmServer
wait_cool = _oom_probe.wait_cool
wait_gpu_free = _oom_probe.wait_gpu_free

# "model weights take 5.79GiB; non_torch_memory takes 0.05GiB; PyTorch activation peak memory
#  takes 2.52GiB; the rest of the memory reserved for KV Cache is 4.74GiB."
_PROFILE = re.compile(
    r"model weights take (?P<weights>[\d.]+)GiB.*?"
    r"non_torch_memory takes (?P<non_torch>[\d.]+)GiB.*?"
    r"activation peak memory takes (?P<activation>[\d.]+)GiB.*?"
    r"reserved for KV Cache is (?P<kv>[\d.]+)GiB"
)
_CONCURRENCY = re.compile(r"Maximum concurrency for (?P<ctx>\d+) tokens per request: (?P<x>[\d.]+)x")

# Seqs sustainable per GiB of KV pool, under the standard mixed workload, before occupancy pins
# at ~99%. Anchored on experiment C: a 1.63 GiB pool went KV-bound at N ~ 125 (spec_on peaked at
# 123/125/127 over three trials). Deliberately the conservative of the two available anchors —
# experiment E's k7 arm reached 149 on the same pool, which would give 91.
#
# This is a planning heuristic for choosing a cap, not a result. It is linear in KV size, which
# holds only while the context-length distribution is fixed, and it is calibrated at one point.
SEQS_PER_KV_GIB = 125.0 / 1.63

# The cap chosen for the C re-run has to clear two bars at once: strictly above 256, so that
# "died at N=256" can be attributed to the scorer rather than to the cap, and with enough KV left
# that the engine can actually *reach* 256 before occupancy binds. The margin keeps the KV-bound
# ceiling comfortably clear of the scorer boundary rather than merely past it.
SCORER_BOUNDARY_N = 256
KV_HEADROOM_MARGIN = 1.20


def probe_cap(cap: int, k: int, log_path: Path, timeout_s: float = 300.0) -> dict:
    """Start a server, capture the memory split, and stop before any request is served."""
    cfg = {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "server": {
            "dtype": "float16",
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": cap,
            "speculative_config": {
                "method": "ngram",
                "num_speculative_tokens": k,
                "prompt_lookup_min": 2,
                "prompt_lookup_max": 5,
            },
        },
    }
    rec: dict = {"max_num_seqs": cap, "num_speculative_tokens": k, "log": str(log_path)}

    wait_gpu_free()
    wait_cool(77.0)

    server = VllmServer(cfg, log_path, port=8000)
    # Not server.start(): that blocks until /health, which includes ~60 s of cudagraph capture
    # and warmup after the only line this tool needs has already been written.
    server.spawn()
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not server.alive():
                rec["error"] = f"server exited during startup (rc={server.returncode()})"
                break
            text = log_path.read_text(errors="replace") if log_path.exists() else ""
            m = _PROFILE.search(text)
            if m:
                rec.update({k2: float(v) for k2, v in m.groupdict().items()})
                c = _CONCURRENCY.search(text)
                if c:
                    rec["max_concurrency_x"] = float(c.group("x"))
                    rec["ctx_tokens"] = int(c.group("ctx"))
                rec["est_kv_bound_n"] = round(rec["kv"] * SEQS_PER_KV_GIB)
                break
            time.sleep(2.0)
        else:
            rec["error"] = f"no profiling line within {timeout_s:.0f}s"
    finally:
        server.stop()
        wait_gpu_free()
    return rec


def recommend_cap(rows: list[dict]) -> dict:
    """Pick the C re-run cap: the largest one above 256 that can still reach N=256."""
    need_kv = SCORER_BOUNDARY_N * KV_HEADROOM_MARGIN / SEQS_PER_KV_GIB
    usable = [
        r for r in rows
        if not r.get("error")
        and r["max_num_seqs"] > SCORER_BOUNDARY_N
        and r.get("kv", 0.0) >= need_kv
    ]
    out = {
        "required_kv_gib": round(need_kv, 3),
        "rule": (
            f"cap > {SCORER_BOUNDARY_N} so the cap cannot explain a death at N={SCORER_BOUNDARY_N}, "
            f"and KV large enough for an estimated {KV_HEADROOM_MARGIN:.2f}x{SCORER_BOUNDARY_N} "
            f"= {SCORER_BOUNDARY_N * KV_HEADROOM_MARGIN:.0f} concurrent sequences"
        ),
    }
    if usable:
        best = max(usable, key=lambda r: r["max_num_seqs"])
        out["cap"] = best["max_num_seqs"]
        out["kv_gib"] = best["kv"]
        out["est_kv_bound_n"] = best["est_kv_bound_n"]
    else:
        # Every cap above 256 costs more KV than the scorer boundary can afford, which is itself
        # the finding: on this GPU the two constraints cannot be separated by raising the cap.
        # Fall back to the smallest step up so the run still produces a data point, and say so.
        out["cap"] = None
        out["note"] = (
            "no cap above 256 leaves enough KV to reach N=256 — on this GPU the scorer boundary "
            "and the sequence cap cannot be separated by raising the cap alone, and the k-sweep "
            "at cap=256 is the only remaining route to decoupling them"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/expF/calibration.json")
    ap.add_argument("--caps", default="128,256,384,512,768,1024")
    ap.add_argument("--k-check", default="2,7",
                    help="extra k values to re-measure at cap=256, confirming k-independence")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = out.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    points = [(int(c), 4) for c in args.caps.split(",") if c.strip()]
    points += [(256, int(k)) for k in args.k_check.split(",") if k.strip()]

    rows = []
    for cap, k in points:
        print(f"[calib] cap={cap} k={k} ...", flush=True)
        rec = probe_cap(cap, k, logs / f"calib_cap{cap}_k{k}.log")
        rows.append(rec)
        if rec.get("error"):
            print(f"[calib]   ERROR: {rec['error']}", flush=True)
        else:
            print(f"[calib]   activation={rec['activation']}GiB kv={rec['kv']}GiB "
                  f"conc={rec.get('max_concurrency_x')}x est_kv_bound_N~{rec['est_kv_bound_n']}",
                  flush=True)

    rec_cap = recommend_cap([r for r in rows if r["num_speculative_tokens"] == 4])
    payload = {"points": rows, "recommendation": rec_cap, "seqs_per_kv_gib": SEQS_PER_KV_GIB}
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n" + "=" * 78)
    print("CAP CALIBRATION")
    print("=" * 78)
    print(f"{'cap':>6} {'k':>3} {'weights':>8} {'activ':>7} {'KV':>7} {'conc':>7} {'est N_kv':>9}")
    for r in rows:
        if r.get("error"):
            print(f"{r['max_num_seqs']:>6} {r['num_speculative_tokens']:>3}   {r['error']}")
            continue
        print(f"{r['max_num_seqs']:>6} {r['num_speculative_tokens']:>3} {r['weights']:>8.2f} "
              f"{r['activation']:>7.2f} {r['kv']:>7.2f} "
              f"{r.get('max_concurrency_x', float('nan')):>6.2f}x {r['est_kv_bound_n']:>9}")
    print()
    print("k-independence check at cap=256: the activation peak must be identical across k,")
    print("because vLLM's profiling pass does not run the speculative scorer. If it is, the")
    print("scorer tensor is provably outside the reservation that bounds vLLM's memory use.")
    print()
    print(f"C re-run cap: {rec_cap['cap']}   ({rec_cap['rule']})")
    if rec_cap.get("note"):
        print(f"  NOTE: {rec_cap['note']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
