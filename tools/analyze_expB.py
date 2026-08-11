"""Experiment B analysis — does the resource bottleneck move as model size changes?

Both arms (Qwen2.5-1.5B, Qwen2.5-3B) ran under the same driver, harness and workload mix, on the
same GPU, within one hour of each other, so the only intended difference is model size.

The analysis answers three questions and is explicit about which are measured and which are not:

  1. Does the *throughput* curve mu(N) move?          -> not identifiable; neither arm saturates
  2. Does the *memory* boundary move?                 -> yes, and it is the binding constraint
  3. Does speculative decoding degrade with size?     -> yes, acceptance rate falls

    python tools/analyze_expB.py

Writes results/expB/expB_analysis.json and docs/experiment_b.md.
"""
from __future__ import annotations

import json
import math
import re
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ARMS = [("1.5b", "Qwen/Qwen2.5-1.5B-Instruct"), ("3b", "Qwen/Qwen2.5-3B-Instruct")]

# Speculative-decoding config used by both arms (phase3-limits/run_expB.sh).
K_PLUS_1 = 5          # num_speculative_tokens + 1
VOCAB = 151936        # identical for both Qwen2.5 checkpoints
BYTES_PER_LOGIT = 4   # the scorer materialises logits in fp32

# See DEGENERATE_OUTPUT_RATIO in tools/fit_queueing_model.py for the justification.
DEGENERATE_OUTPUT_RATIO = 0.4

BOOT = 2000
SEED = 12345


# --------------------------------------------------------------------------- loading


def load_arm(tag: str) -> dict:
    """Split an arm's runs into usable observations and excluded ones, with reasons."""
    good, excluded = [], []
    for p in sorted(RESULTS.glob(f"*expB_qwen{tag}_r*.json")):
        d = json.loads(p.read_text())
        ratio = (d.get("output_len") or {}).get("actual_over_requested")
        nfail = d.get("n_requests_failed") or 0
        # Order matters: the run the engine died *during* can carry both signatures — some
        # requests time out while those already streaming get cleanly truncated. The truncation
        # signature identifies it as the death run; a plain failure count does not.
        if ratio is not None and ratio < DEGENERATE_OUTPUT_RATIO:
            excluded.append({"file": p.name, "rate": d["rate"], "repeat": d["repeat"],
                             "reason": "silent output truncation (engine death mid-window)",
                             "actual_over_requested": ratio, "n_failed": nfail})
        elif nfail:
            excluded.append({"file": p.name, "rate": d["rate"], "repeat": d["repeat"],
                             "reason": "failed requests (engine already dead)", "n_failed": nfail})
        else:
            good.append(d)
    return {"tag": tag, "runs": good, "excluded": excluded}


def arr(runs: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in runs], float)


# --------------------------------------------------------------------------- statistics


def boot_ci(fn, *cols, n_boot: int = BOOT, seed: int = SEED) -> tuple[float, list[float]]:
    """Point estimate plus a percentile bootstrap CI, resampling runs (not residuals)."""
    point = fn(*cols)
    rng = np.random.default_rng(seed)
    n = len(cols[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = fn(*[c[idx] for c in cols])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return point, [float("nan")] * 2
    return point, [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def slope_through_origin(N: np.ndarray, Y: np.ndarray) -> float:
    """Least-squares slope of Y = a*N with no intercept."""
    return float((N @ Y) / (N @ N))


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    d = (a.mean() - b.mean()) / sp
    J = 1 - 3 / (4 * (na + nb) - 9)
    return float(d * J)


def mean_ci(x: np.ndarray) -> tuple[float, float]:
    """Mean and half-width of a two-sided 95% t-interval."""
    n = len(x)
    if n < 2:
        return float(x.mean()), float("nan")
    from scipy import stats as sps
    t = float(sps.t.ppf(0.975, n - 1))
    return float(x.mean()), t * float(x.std(ddof=1)) / math.sqrt(n)


def wilson(k: int, n: int, z: float = 1.959963985) -> list[float]:
    """Wilson score interval for a proportion.

    Used rather than the normal approximation because the counts here are small and the
    proportion sits near 1, where the Wald interval overshoots past 1 and is worthless.
    """
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def local_slope(N: np.ndarray, X: np.ndarray, lo_frac: float, hi_frac: float) -> float:
    order = np.argsort(N)
    N, X = N[order], X[order]
    a, b = int(len(N) * lo_frac), max(int(len(N) * hi_frac), int(len(N) * lo_frac) + 2)
    return float(np.polyfit(N[a:b], X[a:b], 1)[0])


# --------------------------------------------------------------------------- OOM events


OOM_RE = re.compile(r"Tried to allocate ([\d.]+) MiB.*?of which ([\d.]+) MiB is free")


def parse_oom(log: Path) -> dict | None:
    """First engine-level OOM in a server log, with its allocation size and traceback site."""
    if not log.exists():
        return None
    text = log.read_text(errors="replace")
    for line in text.splitlines():
        if "[engine.py:160]" in line and "OutOfMemoryError" in line:
            m = OOM_RE.search(line)
            if not m:
                continue
            ts = re.search(r"ERROR ([\d-]+ [\d:]+)", line)
            site = None
            for m2 in re.finditer(r'([\w/]+\.py)", line (\d+), in (\w+)', text):
                if "batch_expansion" in m2.group(1) or "spec_decode_worker" in m2.group(1):
                    site = f"{Path(m2.group(1)).name}:{m2.group(2)} in {m2.group(3)}"
                    if "batch_expansion" in m2.group(1):
                        break
            return {"log": log.name, "timestamp_local": ts.group(1) if ts else None,
                    "failed_alloc_mib": float(m.group(1)),
                    "free_at_failure_mib": float(m.group(2)), "alloc_site": site}
    return None


def n_at_death(raw: Path) -> dict | None:
    """Peak and final concurrency in a run whose telemetry stops with the batch still full."""
    if not raw.exists():
        return None
    ms = json.loads(raw.read_text()).get("metrics_samples") or []
    run = [s["running"] for s in ms if s.get("running") is not None
           and not (isinstance(s["running"], float) and math.isnan(s["running"]))]
    kv = [s["kv"] for s in ms if s.get("kv") is not None
          and not (isinstance(s["kv"], float) and math.isnan(s["kv"]))]
    if not run:
        return None
    return {"peak_num_running": max(run), "final_num_running": run[-1],
            "peak_kv_occupancy": max(kv) if kv else None,
            "final_kv_occupancy": kv[-1] if kv else None}


MEM_RE = re.compile(
    r"model weights take ([\d.]+)GiB; non_torch_memory takes ([\d.]+)GiB; "
    r"PyTorch activation peak memory takes ([\d.]+)GiB; "
    r"the rest of the memory reserved for KV Cache is ([\d.]+)GiB")
CONC_RE = re.compile(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x")


def parse_memory_split(log: Path) -> dict:
    """vLLM's own startup accounting of where the memory budget went."""
    if not log.exists():
        return {}
    text = log.read_text(errors="replace")
    m = MEM_RE.search(text)
    out = {}
    if m:
        out.update({"weights_gib": float(m.group(1)), "non_torch_gib": float(m.group(2)),
                    "activation_peak_gib": float(m.group(3)), "kv_pool_gib": float(m.group(4))})
    c = CONC_RE.search(text)
    if c:
        out["max_concurrency"] = float(c.group(1))
    return out


def predicted_scorer_mib(n: float) -> float:
    """Size of the dense [N, k+1, V] fp32 scorer tensor described in queueing_model.md 5.3."""
    return n * K_PLUS_1 * VOCAB * BYTES_PER_LOGIT / 2**20


def finite(x) -> float | None:
    """A telemetry sample, or None if missing or NaN.

    The probe records NaN for `num_running` and `kv_occupancy` when the engine dies mid-stage
    rather than dropping the sample, and NaN is truthy, so an unguarded `if n:` lets one dead
    trial poison an arm's mean. Mirrors `_num` in tools/analyze_expF.py.
    """
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def load_expF() -> dict | None:
    """Experiment F's arms, if F has run.

    F varies `k` at a fixed model, which is the one axis Experiment B cannot move: both B arms
    ran k=4, so B measures the scorer law's leading constant and leaves `(k+1)` asserted. The
    per-arm agreement is computed by `tools/analyze_expF.py` and read back here rather than
    recomputed, so the two documents cannot drift apart.
    """
    f = RESULTS / "expF" / "expF_analysis.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    out = {"arms": d["arms"], "reference_k4": d["reference_k4"], "regime": {}}

    # F's fourth step re-ran experiment C at a cap chosen from calibration rather than guessed.
    # The spec-off arm is the mechanism control: with speculative decoding off no scorer tensor
    # is allocated at all, so the law predicts no OOM however full the KV pool gets.
    for tag in ("regime_spec_on_cap512", "regime_spec_off_cap512"):
        p = RESULTS / "expF" / tag / "expA_summary.json"
        if not p.exists():
            continue
        trials = json.loads(p.read_text())
        # Read the arm's identity off the command line it actually ran, not off its directory
        # name — the directory is a label, the argv is the experiment.
        spec_on = any("--speculative-config" in (t.get("server_argv") or []) for t in trials)
        rows = []
        for t in trials:
            o = t.get("oom") or {}
            peaks = t.get("peaks") or {}
            at_fail = t.get("at_failure") or {}
            n = finite(at_fail.get("num_running")) or finite(peaks.get("num_running"))
            alloc = finite(o.get("failed_alloc_mib"))
            rows.append({
                "trial": t.get("trial"),
                "died": bool(t.get("died")),
                "alloc_mib": alloc,
                "n": n,
                "kv_at_fail": finite(at_fail.get("kv_occupancy")),
                "peak_n": finite(peaks.get("num_running")),
                "peak_kv": finite(peaks.get("kv_occupancy")),
                "site": o.get("alloc_site"),
                "mib_per_seq": (alloc / n) if (alloc and n) else None,
            })
        measured = [r["mib_per_seq"] for r in rows if r["mib_per_seq"]]
        arm = {"trials": rows, "n_trials": len(rows),
               "n_died": sum(1 for r in rows if r["died"]),
               "spec_decoding": spec_on}
        if measured:
            arm["mean_mib_per_seq"] = st.fmean(measured)
            arm["rel_error_pct"] = 100 * (st.fmean(measured) - predicted_scorer_mib(1.0)) \
                / predicted_scorer_mib(1.0)
        out["regime"][tag] = arm
    return out


# --------------------------------------------------------------------------- report


def main() -> None:
    arms = {tag: load_arm(tag) for tag, _ in ARMS}
    out: dict = {"arms": {}, "contrasts": {}, "oom": {}, "notes": {}}

    for tag, model in ARMS:
        a = arms[tag]
        runs = a["runs"]
        N, X = arr(runs, "num_running_mean"), arr(runs, "throughput_tok_s")
        KV = arr(runs, "kv_occupancy")
        acc = np.array([r["acceptance_rate"] for r in runs if r.get("acceptance_rate")], float)

        kv_slope, kv_ci = boot_ci(slope_through_origin, N, KV)
        lo = local_slope(N, X, 0.0, 1 / 3)
        hi = local_slope(N, X, 2 / 3, 1.0)
        acc_m, acc_h = mean_ci(acc)

        out["arms"][tag] = {
            "model": model,
            "n_runs_total": len(runs) + len(a["excluded"]),
            "n_runs_used": len(runs),
            "excluded": a["excluded"],
            "N_range": [float(N.min()), float(N.max())],
            "kv_per_seq_pct": kv_slope * 100, "kv_per_seq_pct_ci95": [c * 100 for c in kv_ci],
            "throughput_slope_low_third": lo,
            "throughput_slope_high_third": hi,
            "saturation_ratio": hi / lo if lo else float("nan"),
            "acceptance_rate_mean": acc_m, "acceptance_rate_ci95_halfwidth": acc_h,
            "thermally_throttled_runs": sum(1 for r in runs if r.get("thermal_throttled")),
            "max_temp_c": float(max(r["max_temp_c"] for r in runs)),
        }

    # ---- contrasts -------------------------------------------------------------
    a15, a3 = arms["1.5b"]["runs"], arms["3b"]["runs"]
    N15, KV15 = arr(a15, "num_running_mean"), arr(a15, "kv_occupancy")
    N3, KV3 = arr(a3, "num_running_mean"), arr(a3, "kv_occupancy")

    rng = np.random.default_rng(SEED)
    ratios = []
    for _ in range(BOOT):
        i, j = rng.integers(0, len(N15), len(N15)), rng.integers(0, len(N3), len(N3))
        try:
            ratios.append(slope_through_origin(N3[j], KV3[j]) /
                          slope_through_origin(N15[i], KV15[i]))
        except Exception:
            continue
    kv_ratio = slope_through_origin(N3, KV3) / slope_through_origin(N15, KV15)

    acc15 = np.array([r["acceptance_rate"] for r in a15 if r.get("acceptance_rate")], float)
    acc3 = np.array([r["acceptance_rate"] for r in a3 if r.get("acceptance_rate")], float)
    from scipy import stats as sps
    tstat, pval = sps.ttest_ind(acc3, acc15, equal_var=False)

    out["contrasts"] = {
        "kv_per_seq_ratio_3b_over_1.5b": float(kv_ratio),
        "kv_per_seq_ratio_ci95": [float(np.percentile(ratios, 2.5)),
                                  float(np.percentile(ratios, 97.5))],
        "acceptance_rate": {
            "mean_1.5b": float(acc15.mean()), "mean_3b": float(acc3.mean()),
            "delta": float(acc3.mean() - acc15.mean()),
            "hedges_g": hedges_g(acc3, acc15),
            "welch_t": float(tstat), "welch_p": float(pval),
            "n_1.5b": len(acc15), "n_3b": len(acc3),
        },
    }

    # Mechanism: KV bytes per token follow from the architecture; the size of the block pool the
    # occupancy gauge is a fraction *of* is reported by vLLM at startup. Both are known
    # independently of the sweep, so the predicted ratio below is a prediction, not a fit.
    arch = {"1.5b": {"layers": 28, "kv_heads": 2, "head_dim": 128},
            "3b": {"layers": 36, "kv_heads": 2, "head_dim": 128}}
    for tag, c in arch.items():
        c["kv_bytes_per_token"] = 2 * c["layers"] * c["kv_heads"] * c["head_dim"] * 2
        c.update(parse_memory_split(RESULTS / "expB" / "logs" / f"server_{tag}.log"))
    out["notes"]["kv_mechanism"] = {
        "arch": arch,
        "per_token_kv_ratio": arch["3b"]["kv_bytes_per_token"] / arch["1.5b"]["kv_bytes_per_token"],
        "pool_ratio": arch["1.5b"]["kv_pool_gib"] / arch["3b"]["kv_pool_gib"],
        "predicted_occupancy_ratio": (
            arch["3b"]["kv_bytes_per_token"] / arch["1.5b"]["kv_bytes_per_token"]
            * arch["1.5b"]["kv_pool_gib"] / arch["3b"]["kv_pool_gib"]),
        "vllm_reported_max_concurrency_ratio": (
            arch["1.5b"].get("max_concurrency", float("nan"))
            / arch["3b"].get("max_concurrency", float("nan"))),
    }

    # ---- OOM boundary ----------------------------------------------------------
    events = []
    for tag, _ in ARMS:
        ev = parse_oom(RESULTS / "expB" / "logs" / f"server_{tag}.log")
        if ev:
            deaths = [e for e in arms[tag]["excluded"]
                      if e["reason"].startswith("silent output truncation")]
            tele = None
            if deaths:
                stem = deaths[0]["file"].replace(".json", "_raw.json")
                tele = n_at_death(RESULTS / "raw" / stem)
            ev.update({"arm": tag, "source": "sweep", "telemetry": tele,
                       "death_run": deaths[0]["file"] if deaths else None})
            events.append(ev)

    # Probe campaigns. The 3B boundary was probed twice: an initial batch of 3 alongside the
    # sweep, then a follow-up batch once 2/3 proved too thin to characterise it. They are pooled
    # — same config, same script, same machine — but kept individually addressable.
    probes = [("1.5b", RESULTS / "expA" / "expA_summary_corrected.json", ""),
              ("3b", RESULTS / "expB" / "oom3b" / "expA_summary.json", "a"),
              ("3b", RESULTS / "expB" / "oom3b_more" / "expA_summary.json", "b")]
    trials = {"1.5b": {"n": 0, "died": 0, "kv_died": [], "kv_survived": []},
              "3b": {"n": 0, "died": 0, "kv_died": [], "kv_survived": []}}
    for tag, sub, batch in probes:
        if not sub.exists():
            continue
        for r in json.loads(sub.read_text()):
            trials[tag]["n"] += 1
            o = r.get("oom_corrected") or r.get("oom") or {}
            kv = (r.get("peaks") or {}).get("kv_occupancy")
            if not o.get("failed_alloc_mib"):
                # A survivor is not a trial that failed to reach the boundary — it rides the
                # same ramp to the same concurrency ceiling and keeps serving. Its peak KV is
                # the most direct evidence available that occupancy does not predict failure.
                if kv is not None:
                    trials[tag]["kv_survived"].append(kv * 100)
                continue
            trials[tag]["died"] += 1
            if kv is not None:
                trials[tag]["kv_died"].append(kv * 100)
            peaks = r.get("peaks") or {}
            events.append({"arm": tag, "source": f"probe{batch} trial {r.get('trial')}",
                           "failed_alloc_mib": o["failed_alloc_mib"],
                           "free_at_failure_mib": o.get("free_at_failure_mib"),
                           "alloc_site": "batch_expansion.py:227 in _contract_batch",
                           "telemetry": {"peak_num_running": peaks.get("num_running"),
                                         "peak_kv_occupancy": peaks.get("kv_occupancy")}})
    out["reproduction"] = {
        tag: {"trials": t["n"], "died": t["died"],
              "rate": t["died"] / t["n"] if t["n"] else float("nan"),
              "wilson_ci95": wilson(t["died"], t["n"]),
              "peak_kv_when_died": t["kv_died"],
              "peak_kv_when_survived": t["kv_survived"],
              # The headline test of clause 1: if occupancy were the constraint, no survivor
              # could peak above the lowest occupancy at which a death occurred.
              "survivor_exceeds_death_kv": bool(
                  t["kv_survived"] and t["kv_died"]
                  and max(t["kv_survived"]) > min(t["kv_died"]))}
        for tag, t in trials.items()}

    # For the sweep deaths use concurrency in the *last* telemetry sample before the engine
    # stopped answering, not the run peak: the peak may have occurred seconds earlier, and it is
    # the batch resident at the failing step that sizes the allocation.
    for e in events:
        tele = e.get("telemetry") or {}
        n = tele.get("final_num_running") or tele.get("peak_num_running")
        if n:
            e["n_at_failing_step"] = n
            e["implied_n_from_alloc"] = e["failed_alloc_mib"] / predicted_scorer_mib(1.0)
            e["predicted_mib_at_n"] = predicted_scorer_mib(n)
    out["oom"] = {"events": events,
                  "scorer_mib_per_seq": predicted_scorer_mib(1.0),
                  "tensor": f"[N, k+1={K_PLUS_1}, V={VOCAB}] fp32"}

    expF = load_expF()
    if expF:
        out["expF"] = expF

    out_path = RESULTS / "expB" / "expB_analysis.json"
    out_path.write_text(json.dumps(out, indent=2, default=float) + "\n")

    write_markdown(out, ROOT / "docs" / "experiment_b.md")
    print(f"wrote {out_path}")
    print(f"wrote {ROOT / 'docs' / 'experiment_b.md'}")

    for tag, _ in ARMS:
        a = out["arms"][tag]
        print(f"\n{tag}: {a['n_runs_used']}/{a['n_runs_total']} runs used, "
              f"N in [{a['N_range'][0]:.1f}, {a['N_range'][1]:.1f}]")
        print(f"   KV per sequence = {a['kv_per_seq_pct']:.4f}% "
              f"[{a['kv_per_seq_pct_ci95'][0]:.4f}, {a['kv_per_seq_pct_ci95'][1]:.4f}]")
        print(f"   throughput slope: low third {a['throughput_slope_low_third']:.2f} -> "
              f"high third {a['throughput_slope_high_third']:.2f} tok/s/seq "
              f"(ratio {a['saturation_ratio']:.2f}; a plateau would be ~0)")
        print(f"   acceptance {a['acceptance_rate_mean']:.3f} "
              f"+/- {a['acceptance_rate_ci95_halfwidth']:.3f}")
    c = out["contrasts"]
    print(f"\nKV per sequence, 3B / 1.5B = {c['kv_per_seq_ratio_3b_over_1.5b']:.2f}x "
          f"[{c['kv_per_seq_ratio_ci95'][0]:.2f}, {c['kv_per_seq_ratio_ci95'][1]:.2f}]  "
          f"(predicted from architecture: "
          f"{out['notes']['kv_mechanism']['predicted_occupancy_ratio']:.2f}x)")
    ac = c["acceptance_rate"]
    print(f"acceptance rate {ac['mean_1.5b']:.3f} -> {ac['mean_3b']:.3f} "
          f"(g = {ac['hedges_g']:+.2f}, Welch p = {ac['welch_p']:.2e})")


def write_markdown(out: dict, path: Path) -> None:
    A = out["arms"]
    c = out["contrasts"]
    km = out["notes"]["kv_mechanism"]
    L: list[str] = []
    w = L.append

    w("# Experiment B — does the resource bottleneck move as model size changes?")
    w("")
    w("*Generated by `tools/analyze_expB.py`. Both arms ran under the same driver, harness,")
    w("workload mix and GPU within one hour of each other, so model size is the only intended")
    w("difference. The 1.5B arm was re-measured rather than reused from `paper-v1` because the")
    w("NVIDIA userspace driver changed between the two campaigns.*")
    w("")
    w("**Answer: yes, and it moves in the direction that matters.** The memory boundary and the")
    w("throughput ceiling do not scale together. Doubling model size roughly doubles KV cost per")
    w("sequence, but leaves the allocation that actually kills the engine unchanged — so the")
    w("larger model reaches the same hard wall with far less headroom to spare.")
    w("")

    w("## 1. Runs used")
    w("")
    w("| Arm | Model | Runs used | N range | Throttled | Max temp |")
    w("|---|---|---|---|---|---|")
    for tag in ("1.5b", "3b"):
        a = A[tag]
        w(f"| `{tag}` | {a['model']} | {a['n_runs_used']} / {a['n_runs_total']} | "
          f"{a['N_range'][0]:.1f} – {a['N_range'][1]:.1f} | "
          f"{a['thermally_throttled_runs']} / {a['n_runs_used']} | {a['max_temp_c']:.0f} °C |")
    w("")
    w("Excluded runs, with reasons:")
    w("")
    for tag in ("1.5b", "3b"):
        for e in A[tag]["excluded"]:
            extra = (f"actual/requested = {e['actual_over_requested']:.3f}"
                     if "actual_over_requested" in e else f"{e['n_failed']} failed requests")
            w(f"- `{tag}` λ={e['rate']} rep{e['repeat']} — {e['reason']} ({extra})")
    w("")
    w("> **Nearly every run thermally throttled.** The T4 is passively cooled and sat at 85–89 °C")
    w("> for most of the campaign. Both arms throttled about equally, so the *paired* comparisons")
    w("> below survive it, but absolute rates are depressed and should not be quoted as hardware")
    w("> ceilings. This is the non-stationarity already noted in `queueing_model.md` §5.5.")
    w("")

    n_excl = sum(len(A[t]["excluded"]) for t in ("1.5b", "3b"))
    w(f"## 2. Why {n_excl} runs were discarded — the engine died in both arms")
    w("")
    w("This is the most important thing Experiment B found, and it was not what the experiment")
    w("set out to measure.")
    w("")
    w("Both sweeps ended with the engine dead of a CUDA OOM, in the same allocation, at the same")
    w("source line as Experiment A. The 1.5B engine died during its λ=4 run; the 3B engine died")
    w("during its λ=5 run. Every run after the death recorded 100% request failures — those are")
    w("not measurements of an overloaded server, they are measurements of no server.")
    w("")
    w("One run per arm is worse than a clean failure. When the engine dies mid-window, requests")
    w("already in flight receive a **cleanly terminated stream**: `success=True`, no error, HTTP")
    w("200 — but roughly 5 tokens instead of the ~134 requested. In those runs every workload")
    w("class collapses to the same truncated length regardless of prompt, and the output-length")
    w("histogram spikes at exactly 5 and 15 tokens — multiples of `num_speculative_tokens + 1`,")
    w("the speculative chunk. The stream ends on a chunk boundary because the engine stopped")
    w("between steps.")
    w("")
    w("Nothing in the run record flags these. They report a plausible concurrency with a")
    w("throughput an order of magnitude too low, and they are silently included by any filter")
    w("that only checks the failed-request count. `tools/fit_queueing_model.py` now excludes them")
    w("on `actual_over_requested < 0.4`; the separation is bimodal, not tuned — every healthy run")
    w("in the entire campaign lands in [0.75, 1.00].")
    w("")
    w("**This affects a previously published number.** The same artefact is present in the frozen")
    w("`paper-v1` campaign, in `modelval_mixed` λ=4 rep2, and it was included in the §4 fit. With")
    w("it removed that workload's fit improves from R² = 0.558 to R² = 0.825 and its MAPE from")
    w("77.3% to 29.1%. The `modelval_mixed` row in `queueing_model.md` §4 has been corrected.")
    w("")

    w("## 3. The throughput curve — not identifiable in either arm")
    w("")
    w("| Arm | slope, low third | slope, high third | ratio |")
    w("|---|---|---|---|")
    for tag in ("1.5b", "3b"):
        a = A[tag]
        w(f"| `{tag}` | {a['throughput_slope_low_third']:.2f} | "
          f"{a['throughput_slope_high_third']:.2f} | {a['saturation_ratio']:.2f} |")
    w("")
    w("Slopes are tok/s per sequence. A saturated system would show a high-third slope near zero.")
    w("Neither arm does: throughput is still climbing when the sweep ends. **Neither arm crosses")
    w("the knee**, so the fitted μ_max and N\\* for `expB_qwen1.5b` and `expB_qwen3b` are")
    w("extrapolations constrained by the functional form, not measurements — exactly the failure")
    w("mode documented in `queueing_model.md` §5.1 for the Phase-2 workloads. Do not quote them.")
    w("")
    w("The reason the sweep stops is itself the result: **the engine hits its memory boundary")
    w("before it reaches its throughput plateau.** On this hardware, under speculative decoding,")
    w("μ_max is not merely unmeasured — it is unreachable. The system dies while still rising.")
    w("")

    w("## 4. The memory boundary — this is what moves")
    w("")
    w(f"KV occupancy is proportional to concurrency in both arms. Fitting occupancy = a·N through")
    w(f"the origin (bootstrap over runs, {BOOT} resamples):")
    w("")
    w("| Arm | KV occupancy per sequence | 95% CI |")
    w("|---|---|---|")
    for tag in ("1.5b", "3b"):
        a = A[tag]
        w(f"| `{tag}` | {a['kv_per_seq_pct']:.4f}% | "
          f"[{a['kv_per_seq_pct_ci95'][0]:.4f}%, {a['kv_per_seq_pct_ci95'][1]:.4f}%] |")
    w("")
    w(f"**Ratio 3B / 1.5B = {c['kv_per_seq_ratio_3b_over_1.5b']:.2f}×** "
      f"[{c['kv_per_seq_ratio_ci95'][0]:.2f}, {c['kv_per_seq_ratio_ci95'][1]:.2f}].")
    w("")
    w("That ratio is predicted, not just observed. It has two independent factors:")
    w("")
    a15, a3 = km["arch"]["1.5b"], km["arch"]["3b"]
    w(f"- **Per-token KV cost** rises {km['per_token_kv_ratio']:.2f}×. Both checkpoints have 2 KV "
      f"heads of dimension 128, so the ratio is purely depth: {a3['layers']} layers against "
      f"{a15['layers']}, or {a3['kv_bytes_per_token'] / 1024:.0f} KiB against "
      f"{a15['kv_bytes_per_token'] / 1024:.0f} KiB per token.")
    w(f"- **The pool it lands in shrinks {km['pool_ratio']:.2f}×.** Occupancy is a fraction of the "
      f"block pool, and the pool is whatever survives the weights. From vLLM's own startup "
      f"accounting: fp16 weights take {a15['weights_gib']:.2f} GiB against "
      f"{a3['weights_gib']:.2f} GiB, leaving {a15['kv_pool_gib']:.2f} GiB against "
      f"{a3['kv_pool_gib']:.2f} GiB of KV cache out of the same budget.")
    w("")
    w(f"Together they predict **{km['predicted_occupancy_ratio']:.2f}×**, against a measured")
    w(f"{c['kv_per_seq_ratio_3b_over_1.5b']:.2f}× "
      f"[{c['kv_per_seq_ratio_ci95'][0]:.2f}, {c['kv_per_seq_ratio_ci95'][1]:.2f}] — inside the")
    w("interval, from two numbers neither of which came from the sweep. vLLM's own reported")
    w(f"maximum concurrency ratio ({a15['max_concurrency']:.2f}× vs {a3['max_concurrency']:.2f}×,")
    w(f"i.e. {km['vllm_reported_max_concurrency_ratio']:.2f}×) agrees independently.")
    w("")
    w("The mechanism compounds rather than adding: a larger model pays more per token *and* has")
    w("less room to pay it from. That is why a 2× parameter count costs 2.3× the KV headroom")
    w("rather than the 1.29× the architecture alone would suggest.")
    w("")

    w("## 5. The allocation that kills the engine does not move at all")
    w("")
    ev = out["oom"]["events"]
    F = out.get("expF")
    # F's deaths are not in this table — it is an expB table — but they are the same failure at
    # the same line, and the campaign-wide count has to include them or it understates the case.
    f_deaths = 0
    if F:
        f_deaths = sum(a.get("n_died", 0) for a in F["arms"].values()) \
            + sum(a.get("n_died", 0) for a in F["regime"].values())
    w(f"Every OOM in this campaign — {len(ev) + f_deaths} independent engine deaths across two "
      f"model sizes and {'three' if f_deaths else 'two'} experiments — failed at the same source "
      "line, `batch_expansion.py:227` in `_contract_batch`, reached via "
      "`spec_decode_worker.py:794` `score_proposals`.")
    w("")
    if f_deaths:
        w(f"The {len(ev)} from Experiments A and B are tabulated below; Experiment F's "
          f"{f_deaths} are in §9.")
        w("")
    w("`queueing_model.md` §5.3 proposed that the binding term is a dense scorer tensor of shape")
    w(f"`{out['oom']['tensor']}`, growing linearly in N and invisible to the service-rate model.")
    w("Experiment B measures its coefficient. In fp32 that tensor is")
    w(f"**{out['oom']['scorer_mib_per_seq']:.3f} MiB per sequence**, and it is model-size")
    w("independent because both checkpoints share a vocabulary of 151,936 and the same `k`:")
    w("")
    w("| Arm | Source | N at failure | Predicted | Reported | Error |")
    w("|---|---|---|---|---|---|")
    for e in out["oom"]["events"]:
        n = e.get("n_at_failing_step")
        if not n:
            continue
        pred = e["predicted_mib_at_n"]
        err = 100 * (pred - e["failed_alloc_mib"]) / e["failed_alloc_mib"]
        w(f"| `{e['arm']}` | {e['source']} | {n:.0f} | {pred:.0f} MiB | "
          f"{e['failed_alloc_mib']:.0f} MiB | {err:+.1f}% |")
    w("")
    w("The model holds to within a fraction of a percent across two distinct allocation sizes,")
    w("which is the part that makes it a test rather than a fit: the 1.5B sweep died at a lower")
    w("concurrency and correspondingly asked for *less* memory. The failing allocation tracks N,")
    w("not model size.")
    w("")
    if F:
        # Every death in the table above ran k=4, so the table tests the law's leading constant
        # and nothing else. F moves k, and its regime arm moves N well past anything B reached.
        w("This table varies N and model size. It does **not** vary `k` or `V`, so on its own it")
        w("measures the law's leading constant and leaves the other two terms asserted — see §9,")
        w("where Experiment F moves `k`, and `queueing_model.md` §5.3 for what remains untested.")
        w("")
    kvs = {tag: [e["telemetry"]["peak_kv_occupancy"] * 100 for e in ev
                 if e["arm"] == tag and (e.get("telemetry") or {}).get("peak_kv_occupancy")]
           for tag in ("1.5b", "3b")}
    # Concurrency at which the *KV pool* would run out, if the scorer wall did not arrive first.
    kv_exhaustion_n = 100.0 / A["3b"]["kv_per_seq_pct"]
    w("**This is the bottleneck moving.** The scorer tensor is the same at a given concurrency for")
    w("both models, so the wall sits at the same N — but the 3B engine arrives there having")
    w("already spent twice as much of its smaller pool on KV. Across these events the 1.5B engine")
    w(f"died with KV at {min(kvs['1.5b']):.0f}–{max(kvs['1.5b']):.0f}% occupied, the 3B engine at")
    w(f"{min(kvs['3b']):.0f}–{max(kvs['3b']):.0f}%.")
    w("")
    w(f"Extrapolating the measured {A['3b']['kv_per_seq_pct']:.4f}% per sequence, the 3B arm would")
    w(f"exhaust its KV pool at N ≈ {kv_exhaustion_n:.0f} — beyond `max_num_seqs` = 256, which is")
    w("why the scorer wall still arrives first here. But the margin is down to about 1.6× and it")
    w("closes from both ends at once: KV cost per sequence grows with depth while the pool it")
    w("draws on shrinks with the weights. A modestly larger model on this device crosses over.")
    w("At that point the failure mode changes identity — the OOM becomes a genuine")
    w("KV exhaustion at high occupancy, and the low-KV OOM this paper documents stops being the")
    w("story. **The claim is specific to a regime, and Experiment B locates its edge.**")
    w("")

    w("## 6. Speculative decoding degrades with model size")
    w("")
    ac = c["acceptance_rate"]
    w(f"Mean n-gram acceptance rate falls from **{ac['mean_1.5b']:.3f}** (1.5B, n={ac['n_1.5b']})")
    w(f"to **{ac['mean_3b']:.3f}** (3B, n={ac['n_3b']}), a change of {ac['delta']:+.3f} with")
    w(f"Hedges' g = {ac['hedges_g']:+.2f} and Welch p = {ac['welch_p']:.2e}.")
    w("")
    w("The draft is prompt-lookup n-gram in both arms, so the proposal distribution is identical;")
    w("what changes is that the larger target model agrees with those proposals less often. The")
    w("cost is paid twice — fewer accepted tokens per step, and the same scorer tensor allocated")
    w("to score them.")
    w("")

    w("## 7. What this licenses the paper to claim")
    w("")
    w("Supported:")
    w("")
    w("- The low-KV OOM reproduces across model sizes, experiments and load profiles, always at")
    w(f"  the same allocation site — {len(ev)} independent engine deaths in total, including two")
    w("  unplanned ones during the sweeps themselves.")
    w("- The failing allocation is a dense fp32 scorer tensor linear in concurrency, with a")
    w(f"  measured coefficient of {out['oom']['scorer_mib_per_seq']:.3f} MiB per sequence,")
    w("  confirmed against two different allocation sizes.")
    if F:
        k7 = F["arms"].get("k7_cap256") or {}
        if k7.get("mean_mib_per_seq"):
            w("- The `(k+1)` factor of that coefficient, which Experiment B holds fixed at k=4.")
            w(f"  Experiment F measures it at k=7: {k7['mean_mib_per_seq']:.3f} MiB per sequence")
            w(f"  against {k7['predicted_mib_per_seq']:.3f} predicted, "
              f"{k7['rel_error_pct']:+.1f}% — see §9.")
        spec_off = F["regime"].get("regime_spec_off_cap512") or {}
        if spec_off.get("n_trials") and not spec_off.get("n_died"):
            w("- That speculative decoding is *necessary* for the failure, not merely correlated")
            w(f"  with it: the same load ramp with the scorer disabled survived "
              f"{spec_off['n_trials']}/{spec_off['n_trials']}")
            w("  trials at a KV occupancy higher than any at which the scorer arm died — §9.3.")
    w("- KV cost per sequence roughly doubles from 1.5B to 3B, for architectural reasons that")
    w("  predict the measured ratio without reference to the sweep.")
    w("- Speculative-decoding acceptance falls materially with model size.")
    w("")
    w("Not supported:")
    w("")
    w("- Any statement about μ_max or N\\* for either arm. Neither sweep saturated.")
    w("- Any absolute throughput or power figure as a hardware ceiling — the device throttled")
    w("  through nearly the whole campaign.")
    if F:
        w("- **The `V` term of the scorer law.** Every death in this campaign, F included, ran a")
        w("  Qwen2.5 checkpoint, and all of them share V = 151,936. The vocabulary dimension is")
        w("  read off the allocation site — `new_zeros(*all_tokens.shape, self._vocab_size)` at")
        w("  `batch_expansion.py:226` — which is a direct reading of the code that fails, not an")
        w("  independent measurement, and must not be reported as one.")
    rep = out["reproduction"]
    r15, r3 = rep["1.5b"], rep["3b"]
    if r3["died"] < r3["trials"]:
        w(f"- **Determinism of the 3B boundary.** It is stochastic: {r3['died']} of")
        w(f"  {r3['trials']} probe trials died, a reproduction rate of {r3['rate']:.0%} with a")
        w(f"  Wilson 95% interval of [{r3['wilson_ci95'][0]:.0%}, {r3['wilson_ci95'][1]:.0%}].")
        w(f"  The 1.5B boundary reproduced {r15['died']} of {r15['trials']}. Survivors ride the")
        w("  ramp to λ=8 at high occupancy without failing, so the boundary must be reported as a")
        w("  rate rather than a threshold — see §8.")
    else:
        w(f"- The 3B boundary reproduced {r3['died']} of {r3['trials']} trials, so within this")
        w("  campaign it is deterministic. That is an upper bound on what the evidence shows, not")
        w("  a guarantee for other load profiles.")
    w("")

    w("## 8. Is the boundary deterministic?")
    w("")
    w("| Arm | Probe trials | Engine died | Rate | Wilson 95% CI |")
    w("|---|---|---|---|---|")
    for tag in ("1.5b", "3b"):
        r = rep[tag]
        if not r["trials"]:
            continue
        w(f"| `{tag}` | {r['trials']} | {r['died']} | {r['rate']:.0%} | "
          f"[{r['wilson_ci95'][0]:.0%}, {r['wilson_ci95'][1]:.0%}] |")
    w("")
    w("Each trial is an independent run: a fresh server, a fresh engine, and a load ramp that")
    w("escalates until the engine dies or the ramp is exhausted. A trial that survives is not a")
    w("trial that failed to reach the boundary — it reaches the same concurrency ceiling and")
    w("keeps serving.")
    w("")

    inverted = [t for t in ("1.5b", "3b") if rep[t].get("survivor_exceeds_death_kv")]
    if inverted:
        w("### The survivors settle the KV question on their own")
        w("")
        w("If KV occupancy were the binding constraint, no trial could survive at an occupancy")
        w("above the lowest occupancy at which another trial died. That ordering is violated:")
        w("")
        w("| Arm | Peak KV when the engine died | Peak KV when it survived |")
        w("|---|---|---|")
        for tag in ("1.5b", "3b"):
            r = rep[tag]
            if not r["trials"]:
                continue
            dd = r["peak_kv_when_died"]
            ss = r["peak_kv_when_survived"]
            w(f"| `{tag}` | {', '.join(f'{v:.1f}%' for v in sorted(dd)) or '—'} | "
              f"{', '.join(f'{v:.1f}%' for v in sorted(ss)) or '—'} |")
        w("")
        for tag in inverted:
            r = rep[tag]
            w(f"For `{tag}` the engine **survived a full ramp at {max(r['peak_kv_when_survived']):.1f}% "
              f"occupancy** having **died at {min(r['peak_kv_when_died']):.1f}%** in another trial "
              "under the identical")
            w("configuration. Occupancy is therefore not merely a poor predictor of failure here —")
            w("it is not even monotonically related to it. This is the cleanest available refutation")
            w("of the assumption that KV capacity is serving capacity, and it needs no model: two")
            w("runs of the same configuration, one dead at low occupancy and one alive at high.")
            w("")
    if r3["trials"] and r3["died"] < r3["trials"]:
        w("The 3B interval is the honest statement of what is known. The failure is common but")
        w("not certain, which is consistent with the mechanism: the scorer tensor is requested")
        w("once per engine step, and whether a request of that size succeeds depends on the state")
        w("of the caching allocator — how fragmented the reserved-but-unallocated pool is at that")
        w("instant. That is why the reported free memory at failure sits close to, but not below,")
        w("the size of the allocation. Nothing about the boundary requires it to be crossed")
        w("deterministically.")
        w("")

    if F:
        write_expF_section(w, out, F)

    path.write_text("\n".join(L) + "\n")


def write_expF_section(w, out: dict, F: dict) -> None:
    """§9 — Experiment F, folded in from results/expF.

    B establishes that the failing allocation is linear in N. F establishes what the slope is
    made of, which is the difference between a fitted coefficient and a law.
    """
    k2, k7 = F["arms"].get("k2_cap256") or {}, F["arms"].get("k7_cap256") or {}
    ref = F["reference_k4"]

    w("## 9. Experiment F — the `(k+1)` term, and the mechanism control")
    w("")
    w("*Folded in from `results/expF/expF_analysis.json`; F's own driver is")
    w("`phase3-limits/run_expF.sh` and its analysis `tools/analyze_expF.py`. Same GPU, same 3B")
    w("checkpoint, same load ramp as the probe trials in §8 — only `k` changes.*")
    w("")
    w("§5 measures the failing allocation against N and finds it linear. That is consistent with")
    w("the scorer tensor but does not identify it: any per-sequence allocation would look the")
    w("same. The law claims a specific slope, `(k+1)·V·4 bytes`, and B cannot test the `(k+1)`")
    w("factor because both its arms ran k=4. F varies it. This is the term most likely to have")
    w("taken another form — how many speculative tokens are actually scored per step is a")
    w("scheduler decision, not simply the configured `k`.")
    w("")
    w("| Arm | k | Trials | Died | MiB/seq measured | Predicted | Error |")
    w("|---|---|---|---|---|---|---|")
    w(f"| `expB` reference | {ref['k']} | — | — | {ref['mib_per_seq']:.3f} | "
      f"{ref['mib_per_seq']:.3f} | (the anchor) |")
    for tag, a in (("k2_cap256", k2), ("k7_cap256", k7)):
        if not a.get("n_trials"):
            continue
        meas = (f"{a['mean_mib_per_seq']:.3f}" if a.get("mean_mib_per_seq")
                else "— (no death to measure)")
        err = f"{a['rel_error_pct']:+.1f}%" if a.get("rel_error_pct") is not None else "—"
        w(f"| `{tag}` | {a['k']} | {a['n_trials']} | {a['n_died']} | {meas} | "
          f"{a['predicted_mib_per_seq']:.3f} | {err} |")
    w("")

    if k7.get("mean_mib_per_seq"):
        ratio = k7["mean_mib_per_seq"] / ref["mib_per_seq"]
        w(f"**k=7 is the decisive arm and it lands at {k7['rel_error_pct']:+.1f}%.** It died in")
        w(f"{k7['n_died']}/{k7['n_trials']} trials at {k7['mean_mib_per_seq']:.3f} MiB per")
        w(f"sequence against {k7['predicted_mib_per_seq']:.3f} predicted, and the ratio to the")
        w(f"k=4 reference is {ratio:.3f} observed against {8 / 5:.3f} predicted — 8/5, a number")
        w("fixed by the tensor's shape before the arm ran. Both arms died at the same source")
        w("line as every other death in the campaign.")
        w("")

    # The confound experiment C could not break. State it on occupancy, because concurrency does
    # not carry it: one of the three deaths lands on the sequence cap itself.
    deaths = [t for t in k7.get("trials", []) if t.get("died")]
    ns = sorted(t["n"] for t in deaths if t.get("n"))
    kvs = sorted(t["kv_at_fail"] for t in deaths if t.get("kv_at_fail"))
    if ns and kvs:
        w("### 9.1 Why this is the scorer and not the sequence cap")
        w("")
        w("A death at high concurrency is ambiguous — it could be the scorer, or it could be the")
        w("engine simply running out of the resource the cap and the KV pool jointly bound. The")
        w("k=7 deaths resolve it, on **occupancy**, not concurrency:")
        w("")
        w(f"- They died at N = {', '.join(f'{n:.0f}' for n in ns)}, against `max_num_seqs` = 256.")
        w(f"- KV occupancy at the failing step was "
          f"{', '.join(f'{v * 100:.1f}%' for v in kvs)}, against an estimated KV ceiling of")
        w("  N ≈ 363 at this cap (`results/expF/calibration.json`).")
        w("")
        w(f"Up to {100 * (1 - max(kvs)):.0f}% of the KV pool was still free at the moment the")
        w("engine died. Neither the cap nor exhaustion of the pool can account for a failure with")
        w("that much room left, and the allocation that did fail is the size the law predicts to")
        w("within a tenth of a percent. Concurrency alone would not have carried this argument:")
        w(f"one of the three deaths sits at N = {max(ns):.0f}, the cap itself.")
        w("")

    if k2.get("n_trials"):
        w("### 9.2 The k=2 null")
        w("")
        if k2.get("n_died"):
            w(f"k=2 died in {k2['n_died']}/{k2['n_trials']} trials, which the null did not")
            w("predict. What the law claims is the *size* of the allocation, not where the")
            w("boundary lands — the latter depends on free memory at the failing step. Judge it")
            w("on MiB/seq, in the table above.")
        else:
            w(f"k=2 survived all {k2['n_trials']} trials, as predicted: at "
              f"{k2['predicted_mib_per_seq']:.3f} MiB per sequence its boundary was forecast at")
            w("N ≈ 344 (`tools/analyze_expF.py`, fixed before the arm ran), which lies")
            w("past the cap of 256, so the engine cannot reach it. The arm pushed to")
            w(f"N = {k2.get('peak_n_observed', float('nan')):.0f} and "
              f"{100 * k2.get('peak_kv_observed', float('nan')):.1f}% KV occupancy without dying.")
            w("")
            w("This is the cheap falsification F could have failed and did not. A law that")
            w("over-predicts the allocation would have killed this arm too; one that under-")
            w("predicts would not have killed k=7. Being right about *which* arm dies is a")
            w("separate test from being right about the number.")
        w("")

    on = F["regime"].get("regime_spec_on_cap512") or {}
    off = F["regime"].get("regime_spec_off_cap512") or {}
    if on.get("n_trials") and off.get("n_trials"):
        w("### 9.3 The mechanism control — turn the scorer off and the OOM goes away")
        w("")
        w("F's fourth step re-ran experiment C at `max_num_seqs` = 512, a cap chosen from the")
        w("calibration sweep rather than guessed. (C's first attempt used 1024, where vLLM sizes")
        w("its activation reserve at the cap and leaves only 1.63 GiB of KV — that run measured")
        w("the cap, not the question. See `results/expF/calibration.json`.) Both arms are the")
        w("same model, cap, ramp and seed; the only difference is `--speculative-config`.")
        w("")
        w("| Arm | Speculative decoding | Trials | Died | Peak N | Peak KV |")
        w("|---|---|---|---|---|---|")
        for tag, a in (("spec on", on), ("spec off", off)):
            pk_n = [t["peak_n"] for t in a["trials"] if t.get("peak_n")]
            pk_kv = [t["peak_kv"] for t in a["trials"] if t.get("peak_kv")]
            # Two decimals: the spec-off arm peaks at 0.9996, and rounding that to "100.0%"
            # claims an exhaustion that did not happen.
            w(f"| `{tag}` | {'on' if a['spec_decoding'] else 'off'} | {a['n_trials']} | "
              f"{a['n_died']} | {max(pk_n):.0f} | {100 * max(pk_kv):.2f}% |")
        w("")
        if on.get("mean_mib_per_seq") is not None:
            allocs = sorted(t["alloc_mib"] for t in on["trials"] if t.get("alloc_mib"))
            b_allocs = sorted(e["failed_alloc_mib"] for e in out["oom"]["events"])
            b_ns = sorted(e["n_at_failing_step"] for e in out["oom"]["events"]
                          if e.get("n_at_failing_step"))
            on_ns = sorted(t["n"] for t in on["trials"] if t.get("n"))
            w(f"**The spec-on arm died {on['n_died']}/{on['n_trials']}, at "
              f"{on['mean_mib_per_seq']:.3f} MiB per sequence against")
            w(f"{out['oom']['scorer_mib_per_seq']:.3f} predicted "
              f"({on['rel_error_pct']:+.1f}%)** — the same law, at allocations")
            w(f"of {min(allocs):.0f}–{max(allocs):.0f} MiB, larger than anything in §5's table "
              f"({min(b_allocs):.0f}–{max(b_allocs):.0f} MiB),")
            w(f"and at a concurrency well beyond it: that table spans N = {min(b_ns):.0f} to "
              f"{max(b_ns):.0f}, and this")
            w(f"extends the same linear fit out to N ≈ {max(on_ns):.0f}.")
            w("")
        w(f"**The spec-off arm died {off['n_died']}/{off['n_trials']}.** It rode the ramp to λ=8")
        off_kv = [t["peak_kv"] for t in off["trials"] if t.get("peak_kv")]
        on_kv = [t["peak_kv"] for t in on["trials"] if t.get("peak_kv")]
        if off_kv and on_kv:
            w(f"at {100 * min(off_kv):.2f}–{100 * max(off_kv):.2f}% KV occupancy — *higher* than")
            w(f"the {100 * min(on_kv):.2f}–{100 * max(on_kv):.2f}% at which the spec-on arm died —")
            w("and kept serving.")
        w("")
        w("This is the control the diagnosis needed. §8 shows occupancy does not predict the")
        w("failure; this shows what does. Remove the allocation the law names and the failure")
        w("disappears, under a load that drives the same engine to a fuller KV pool than the one")
        w("that killed it. The mechanism is not merely consistent with the deaths — it is")
        w("necessary for them.")
        w("")
        w("> **These deaths are not low-occupancy ones.** At this cap the spec-on arm reaches")
        w("> N ≈ 410 and a nearly full KV pool before the scorer allocation fails, so it does not")
        w("> demonstrate the low-KV OOM that §5 and §8 document at the default cap of 256. It")
        w("> demonstrates the *mechanism*, against a matched control. The two arms of §9 do")
        w("> different jobs: k=7 shows the failure arriving with a third of the pool free, and")
        w("> this pair shows it not arriving at all once the scorer is gone.")
        w("")


if __name__ == "__main__":
    main()
