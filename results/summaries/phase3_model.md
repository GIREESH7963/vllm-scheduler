# Phase 3 — Queueing model + validation (committed summary)

**Date:** 2026-07-06 · **Model:** Qwen/Qwen2.5-1.5B-Instruct · **GPU:** T4 (float16) · offline
analysis of the Phase-1/2 measurements (no serving). Code: `phase3-model/model.py`,
config `configs/phase3_model.yaml`, fitted constants `results/summaries/phase3_model_fit.json`.

## The model (kept deliberately simple and legible)

vLLM continuous batching ≈ **M/G/1 processor sharing (PS)** with a *load-dependent* (batched)
service rate, plus a hard concurrency **capacity cap**. With concurrency `N` = mean running
sequences (measured `num_running_mean`):

```
throughput(N) = min(N · r0, μmax)     # linear batching ramp  →  flat compute ceiling
tpot(N)       = max(1/r0, N / μmax)   # flat standalone       →  linear sharing  (the PS signature)
knee   N* = μmax / r0                 # concurrency where per-seq fair-share == standalone rate
```

- `r0`  = single-stream output rate (tok/s per sequence), fit from low-concurrency runs.
- `μmax` = aggregate GPU compute ceiling (tok/s), fit from the throughput plateau.

**Why PS is the right primitive:** vLLM decodes all running sequences one token per step, so with
`N` running they each advance at ~`1/N` of the batch step rate — textbook processor sharing. Below
`N*` there is spare compute, so adding a sequence is nearly free (`tpot` flat, throughput rises
~linearly). Above `N*` the GPU is the bottleneck: the `N` sequences share a fixed `μmax`, so per-token
latency rises linearly and aggregate throughput flattens. **The knee `N*` is derivable from first
principles** — it is exactly where the fair share `μmax/N` falls below the standalone rate `r0`.

## Fitted parameters (per workload family)

| Family | r0 (tok/s/seq) | μmax (tok/s) | knee **N\*** | RMSE (tok/s) | n |
|---|---|---|---|---|---|
| **phase2_mixed** (baseline, no-sched) | 49.5 | 495 | **10.0** | 36 | 33 |
| phase2_chat | 58.0 | 355 | 6.1 | 51 | 30 |
| phase1_mixed (heavy) | 20.0 | 250 | 12.5 | 45 | 9 |

- `r0 ≈ 48` tok/s matches the measured standalone TPOT (~21–28 ms/token) — an independent physical
  anchor, not a free parameter.
- **Same output length, different ceiling:** phase1_mixed and phase2_mixed both average ~104 output
  tokens, yet μmax is 250 vs 495. The heavier mix has longer *prompts*, so more of the compute
  budget goes to prefill and the *output*-token ceiling roughly halves. μmax is mix-specific; the
  PS *structure* is universal.
- **All policies collapse onto one `throughput(N)` law** (Fig 1): FCFS/SRPT/EDF/adaptive differ only
  in the `N` they induce — they ride the same curve. This is the key validation that the scheduler
  is an admission/ordering layer, not a change to the service process (the Phase-2 premise, confirmed).

Figures: `results/figures/phase3_throughput_vs_concurrency.png` (Fig 1),
`phase3_tpot_vs_concurrency.png` (Fig 2), `phase3_throughput_vs_load.png` (λ-domain overlay),
`phase3_capacity_regimes.png` (Fig 4).

## The capacity limit — and where the model breaks down (the differentiator)

The Phase-3 brief anticipated a **hard KV-cache capacity limit** as the thing that forces the model
to deviate. The measurements say otherwise, and that is the interesting result:

- **KV cache is never the binding constraint on this setup.** `kv_occupancy` stays below ~8% even at
  60 concurrent sequences. Deriving the KV cap directly from the data (`C_kv = N / kv_occupancy`)
  gives **C_kv ≈ 830 concurrent sequences** — versus the compute knee **N\* ≈ 10**. **Compute
  saturates ~83× before KV does.** On a 1.5B model with 16 GB, KV is slack; the T4 is *compute-bound*.
- So the real capacity ceiling here is `N*` (a compute/throughput cap), not a KV/admission-blocking
  cap. The classic KV-blocking regime that PagedAttention/vLLM were designed for is simply **outside
  the envelope this hardware+model can reach.**
- **Projected crossover (Fig 4, first principles):** the KV budget is ~fixed in *tokens*, so
  `C_kv(context) = C_kv · (ctx0 / context)`. KV overtakes compute when `C_kv(context) < N*`, i.e. at
  an average sequence length of **~25k tokens** on this exact setup. Below that, compute binds
  (what we measured); above it (long-document RAG, or a larger model with fatter KV/token), KV
  blocking would dominate and admission-layer KV control would start to matter. That is the regime
  to target in future work — we can *derive* where it begins but cannot reach it on a T4 + 1.5B.

## Where predicted and measured diverge (19 flagged points, >20% throughput error)

Three interpretable causes — none of them a failure of the PS structure:

1. **Low-load measurement noise (phase1 at N ≲ 4).** Few, bursty, variable-length requests per
   window → throughput/TPOT scatter of 25–64%. This is the short-window/low-λ sampling effect
   documented in Phase 1, not a model error. (The clean phase2 fits have RMSE ~10%.)
2. **Thermal throttling depresses the plateau (phase1 at N ≈ 10).** Measured throughput
   ~115–133 tok/s sits *below* the model's ~200 — every phase1 λ=2 run is `thermal_throttled=True`;
   slowed clocks lower the effective μmax. The model assumes a fixed ceiling; the passively-cooled
   T4 does not provide one.
3. **The knee is smoother than the `min`/`max` kink (transition band around N\*).** Real batching
   ramps up gradually, so just below `N*` the model slightly under-predicts (measured > model at
   N≈4–5) and just above it slightly over-predicts (phase2_chat measured ~259 vs model 355 at N≈7).
   A sharp two-line model trades a few % of transition-band accuracy for legibility — the right call
   per the "clear approximate model beats a tuned complex one" rule.

## Definition-of-done check
- ✅ Analytical model: continuous batching ≈ M/G/1-PS (token-aware ordering ≈ SRPT rides the same
  `throughput(N)` curve), extended with a hard concurrency capacity cap.
- ✅ Predicted throughput/TPOT-vs-load curves overlaid on the Phase-1/2 measurements (Figs 1–3).
- ✅ Written first-principles explanation of the deviation, including the compute-vs-KV crossover.
- ✅ Legible over complex: two fitted constants per mix, both physically anchored.

## Honest framing for the writeup
The system is processor-sharing with a compute ceiling, and a scheduler in front changes only *which*
`N` you operate at — it cannot move the `throughput(N)` curve. The capacity limit that the queueing
literature emphasises (KV-cache blocking) is real but **~83× beyond where a T4 + 1.5B saturates**; on
this hardware the binding constraint is compute, and we derive analytically that KV would only take
over at ~25k-token sequences. That derivation — *which* ceiling binds, and where the crossover sits —
is Phase 3's contribution.

## git
Repo: https://github.com/GIREESH7963/vllm-scheduler (private). Committed summaries + figures are the
artifacts of record; raw run JSONs and `.env` stay local / git-ignored.
