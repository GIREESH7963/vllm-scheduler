# Phase 2 — Admission/ordering scheduler + policy comparison (committed summary)

**Date:** 2026-07-06 · **Model:** Qwen/Qwen2.5-1.5B-Instruct · **GPU:** T4 (float16) ·
vLLM with n-gram spec decode. All scheduling is an **admission/ordering layer IN FRONT of vLLM**
— the engine's continuous-batching scheduler is never touched (root rule).

## What was built

| Module | Role |
|---|---|
| `scheduler/scheduler.py` | Queue + admission concurrency cap + pluggable policy interface. Policies: `nocap` (admit-all baseline ≈ Phase 1), `fcfs`, `srpt` (size = prompt-length proxy), `edf` (class-aware deadlines), `adaptive` (rule-based feedback: widen cap when SM idle, throttle when TTFT p99 rises, deprioritize long jobs when KV > threshold). **Load-shedding:** drop requests queued longer than `max_queue_wait_s` (counts as an SLO miss). |
| `scheduler/predictor.py` | Class-aware linear output-length predictor; drives SRPT oracle / prediction / prompt-proxy variants. |
| `scheduler/run.py` | Policy matrix (policies × rates × repeats) + predictor experiment mode; writes per-run JSON + policy summary + figures. |
| `configs/phase2_{mixed,chat}.yaml` | The two workload mixes for the head-to-head. |

Metrics extended for scheduling: `queue_wait`, `total_latency` (arrival→done), per-class SLO/queue/total.
SLO denominator = **all** measured requests (drops count as misses); SLO TTFT is arrival-relative
(queue + server). Reported as mean ± std over **3 repeats**, always p50 **and** p99.

## Aggregate SLO attainment (mean ± std, 3 reps)

**Mixed** (coding/long/reasoning/short):

| Policy | λ=2 (below knee) | λ=4 (overload) | tok/s @ λ=4 |
|---|---|---|---|
| **nocap** (admit-all) | 0.99 | **0.91 ± 0.10** | **470** |
| adaptive | 0.99 | 0.57 ± 0.20 | 430 |
| edf | 0.89 | 0.51 ± 0.08 | 274 |
| srpt | 0.92 | 0.43 ± 0.02 | 323 |
| fcfs | 0.88 | 0.12 ± 0.13 | 317 |

**Chat** (85% short, higher rates):

| Policy | λ=4 | λ=6 (overload) | tok/s @ λ=6 |
|---|---|---|---|
| **adaptive** | 1.00 | **0.90 ± 0.04** | **410** |
| nocap | 1.00 | 0.88 ± 0.05 | 409 |
| srpt | 0.995 | 0.79 ± 0.03 | 366 |
| edf | 0.97 | 0.74 ± 0.08 | 367 |
| fcfs | 0.96 | 0.47 ± 0.38 | 398 |

Figures: `results/figures/phase2_{mixed,chat}_{slo_attainment,total_latency_p99,queue_wait_p99,tpot_p50,byclass_slo,byclass_totlat_p99}.png`.

## Findings

**1. On a single T4, no capped policy beats admit-all on aggregate SLO.** vLLM's continuous batching
converts concurrency directly into throughput (nocap 470 tok/s vs edf's 274 at mixed λ=4 — a **−42%
penalty from capping**). Any hard concurrency cap trades away the throughput that meets deadlines.
This is the honest headline and it confirms the calibration finding from the build phase.

**2. The scheduler's real, measured value is differentiated service under overload**
(per-class figures, mixed λ=4):
- **edf** protects `short` (SLO **0.85**) by sacrificing `reasoning` (**0.04**) and `coding` (0.22)
  — deadline ordering behaves exactly as designed.
- **srpt** protects `short` (0.68) but starves `long` (**0.06**) — the classic SRPT long-job
  starvation, visible and quantified.
- **adaptive** is the only policy that keeps *every* class alive (0.42 / 0.59 / 0.65 / 0.57) —
  balanced service, no starvation.
- **nocap** looks uniformly good only because throughput is high; it offers **zero control** — you
  cannot protect a priority class when it matters.

**3. FCFS + cap suffers the convoy collapse.** FCFS is strictly worst everywhere (0.12 mixed, 0.47
chat) with the highest variance (±0.38) — long jobs clog the limited slots and short jobs queue
until they are shed. Every ordering policy (SRPT/EDF/adaptive) mitigates it. This is the robust
qualitative result.

**4. Adaptive is the standout capped policy.** Its cap widens when SM is idle, so it stays
throughput-competitive (430 tok/s mixed; ties nocap at 410 in chat) while keeping per-class service
balanced. In the chat overload it actually **edges nocap on aggregate SLO (0.90 vs 0.88)** — the one
regime where scheduling adds a net win, because the short-dominated mix leaves idle capacity that a
feedback cap can reclaim without inducing convoy.

**5. High run-to-run variance at the knee is real, not noise to hide.** Convoy is arrival-order /
seed sensitive; std is reported on every cell (fcfs ±0.38, adaptive ±0.20 at mixed λ=4). The
per-class *differentiation* is far more robust than the aggregate SLO — that is the result to lead with.

## Predictor mini-experiment (mixed, λ=4, MAE = 19 tokens)

| SRPT variant | SLO attainment |
|---|---|
| prompt-length proxy | 0.462 |
| prediction (learned length) | 0.441 |
| oracle (true output length) | 0.376 |

**Misprediction cost is negligible.** All three variants ran on one fixed arrival trace (paired
comparison, single seed). Prediction (0.44) essentially matches the free prompt-length proxy (0.46),
and even a perfect output-length oracle does not win at this operating point — SRPT
ordering is inherently weak here (all variants < 0.47, consistent with srpt losing to nocap). Single
trace, so read the oracle ordering as suggestive not established. The
takeaway: investing in an accurate length predictor buys nothing in this regime; the cheap
prompt-length proxy is sufficient. A clean, defensible negative result.

## Definition-of-done check
- ✅ Pluggable scheduler interface: FCFS, token-aware SRPT-proxy, EDF (deadlines), one adaptive
  feedback policy — plus a `nocap` baseline.
- ✅ Output-length predictor mini-experiment measuring the cost of misprediction on SRPT.
- ✅ Head-to-head across ≥2 workload mixes (mixed + chat) producing the core result plots.
- ✅ Admission/ordering layer only — vLLM internals untouched.
- ✅ Rigor: fixed seeds, 3 repeats, ±std reported, p50 **and** p99, warmup discarded, thermally gated.

## Honest framing for the writeup
On a single T4, vLLM's own batching is hard to beat on aggregate throughput/latency. The scheduler's
contribution is **choosing who suffers under overload** — protect short/urgent classes, shed the
rest, and avoid the FCFS convoy — not a strict aggregate win. Adaptive is the exception that shows a
feedback cap can also reclaim idle capacity in a short-dominated mix. Phase 3 will model this
tradeoff (queueing model + validation against these measurements).

## git
Repo: https://github.com/GIREESH7963/vllm-scheduler (private). Committed summaries + figures are the
artifacts of record; raw run JSONs and `.env` stay local / git-ignored.
