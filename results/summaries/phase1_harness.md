# Phase 1 — Workload generator + measurement harness (committed summary)

**Date:** 2026-07-06 · **Model:** Qwen/Qwen2.5-1.5B-Instruct · **GPU:** T4 (float16) ·
vLLM 0.8.5.post1 with n-gram spec decode.

## What was built (the reusable harness every later phase depends on)

| Module | Role |
|---|---|
| `common/workload.py` | Profiles (short/long/reasoning/coding), token-length distributions (fixed/uniform/lognormal), weighted `WorkloadMix`. **Coding = the spec-decode class** (n-gram-lookup-friendly). Seeded → reproducible. |
| `common/loadgen.py` | **Open-loop Poisson** driver: Exp(λ) inter-arrival gaps, dispatched on schedule regardless of in-flight count. Streams each request; records arrival, dispatch, TTFT, per-token times, e2e, actual-vs-requested length, prompt tokens. |
| `common/metrics.py` | `aggregate_run()` → root schema + extras (per-class breakdown, dispatch-lag, queue depths, output-len); warmup discarded by arrival time; telemetry filtered to the measured window via the shared monotonic clock. `summarize_repeats()` → mean±std across repeats. |
| `common/plots.py` | Headless figures: latency-vs-load, throughput-vs-load, utilization-vs-load, with ±1 std error bars over repeats. |
| `phase1-harness/run.py` | Sweeps `arrival_rates × repeats`, one JSON per run + sweep summary + figures. One-time pre-sweep warmup from idle. Optional `save_raw` dumps per-request + per-sample time-series. |
| `configs/phase1_{chat,mixed,rag,specheavy}.yaml` | Four workload mixes; `phase1_specheavy` is the spec-decode-heavy one. `phase1_validation.yaml` = fast fixed-workload stability check. |

## Validation run (`phase1_validation`, 2 rates × 3 repeats, 40s windows)

| λ (req/s) | throughput (tok/s) | TTFT p50/p99 (ms) | SM-active % | acceptance | dispatch-lag p99 |
|---|---|---|---|---|---|
| 1 | 93.6 ± 13.0 | ~200 / ~600 | 49–79 | 0.55–0.62 | < 5 ms |
| 2 | **167.9 ± 8.9** | ~400 / ~1400 | ~82 | 0.64–0.66 | < 9 ms |

Figures: `results/figures/phase1_validation_{latency,throughput,utilization}_vs_load.png`.

## Definition-of-done check
- ✅ Poisson arrivals; configurable prompt/output distributions; 4 mixes incl. a spec-decode class.
- ✅ Async open-loop driver at target λ; per-request timings recorded.
- ✅ One results JSON per run in the root schema; p50 **and** p99; warmup discarded.
- ✅ Plotting helper produces latency/throughput/utilization figures.
- ✅ Rigor: fixed seeds, ≥3 repeats, variance reported (±std + error bars), always p50/p99.

## Findings that carry into Phase 2/3
- **Client is not the bottleneck**: dispatch-lag p99 stayed < 9 ms even at λ=2 — the open-loop
  assumption holds; measured latency is the server's, not the client's.
- **Batching engages under load**: SM-active rose from 71% (single-stream, Phase 0) to ~82%.
- **Capacity is low for this mix (~1 req/s)**: at λ=2 the system is already in deep overload
  (e2e p50 ~11 s, ~21 concurrent running, SLO attainment ~6%). Good saturation data; when
  characterizing the knee, sweep λ ≈ 0.5–2.
- **Throughput variance is a short-window/low-λ sampling effect, not instability.** At λ=2
  (more requests/window) CV≈5%; at λ=1 with 20–30 s measured windows CV can hit 15–45% because
  few, bursty, variable-length requests land in the window. **Guideline: window length must grow
  as λ falls.** The real mix configs use 90 s windows; increase further for λ<1. Variance is
  always reported so this is visible rather than hidden.
- **Actual vs requested output length ≈ 0.87** (model emits EOS before max_tokens); recorded
  per-request — Phase 2 scheduling needs this.

## Not a git repo yet — this summary + committed figures are the artifact. `git init` when ready.
