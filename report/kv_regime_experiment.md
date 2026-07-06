# Experiment: does long context push a single-T4 vLLM into a KV-bound regime?

**Status:** design + protocol (branch `kv-regime-experiment`). Extends the Phase-3 finding that on a
T4 + 1.5B the system is *compute-bound* (compute knee N\*≈10 ≪ derived KV cap C_kv≈830) and KV would
only bind at ~25k-token sequences. This experiment tries to *reach* that regime on the same hardware.

## The question (framed honestly)

Growing prompt context does **two** things at once, so it does not cleanly isolate KV:
1. KV per sequence grows → pushes toward a **KV-bound** ceiling (cache fills, admission blocks).
2. Prefill cost per request explodes → pushes toward a **prefill-compute-bound** ceiling.

So the hypothesis is **not** "long context ⇒ KV-bound." It is:

> **As prompt context grows from ~2k to ~24k tokens, which ceiling binds first — KV capacity or
> prefill compute — and does the achieved concurrency follow the model's `min(N*, C_kv(context))`
> envelope?**

Both outcomes are publishable-flavored:
- **KV binds** (kv_occupancy→1, `num_waiting`>0 while SM<saturation): confirms the derived crossover;
  KV-aware admission should then win (see policy sub-experiment).
- **Prefill compute binds first** (SM saturates, kv_occupancy stays <1): a *different* but honest
  result — "on this GPU, long-context serving is prefill-bound, not KV-bound; KV never becomes the
  scarce resource," which *revises* the naive expectation and is itself a contribution.

## Why the current Phase-3 constants may not transfer

`μmax`, `r0`, and `C_kv` were fit in a **short-prompt, decode-dominated** regime. Prefill has a
different compute profile, so the `throughput(N)` law and even `C_kv` (KV-per-token is constant, but
the *achievable* concurrency is gated by prefill scheduling) may shift. The model's crossover line is
therefore a **prediction to test**, not ground truth — the overlay in `model/model.py`
(`fig_kv_regime_validation`) plots measured points *against* it precisely to expose any gap.

## Protocol

**0. Pin the server (prerequisite — the current server is uncontrolled).** Relaunch vLLM with
explicit, recorded flags so the KV budget is a *known* quantity, not back-derived:
```
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype float16 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}' \
  --max-model-len 32768 --max-num-seqs 64 --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill --port 8000
```
Record `num_gpu_blocks` / KV-token budget from the startup log → this is the *ground-truth* C_kv.

**1. Calibration probe (cheap, do first).** Run one `kvsweep_ctx8k` burst. Confirm (a) we can sustain
saturating concurrency, (b) `kv_occupancy` and `num_waiting` respond, (c) no mass timeouts/shedding.
Adjust the per-context arrival rates if the system trickles instead of saturating. **Do not launch
the full sweep until this passes.**

**2. Context sweep** (`configs/kvsweep_ctx{2k,8k,16k,24k}.yaml`, fixed-ish saturating load per point).
Trace achieved concurrency, kv_occupancy, num_waiting, and SM-active vs context. This produces the
`phase3_kv_regime_validation.png` overlay.

**3. Regime attribution.** At each context classify the bottleneck: KV-bound (kv→1, waiting>0, SM
below its short-context ceiling) vs prefill-compute-bound (SM saturated, kv<1). This is the core
result and it settles the hypothesis.

**4. Policy sub-experiment** (`configs/kvbound_policies.yaml`, only if step 3 shows a KV-bound point).
A 16k-heavy + short/urgent mix under overload, all five policies. Hypothesis: in a KV-bound regime,
KV-aware admission (adaptive deprioritizes long jobs when KV high; edf protects the urgent class)
yields a **measurable urgent-class SLO gain over admit-all** — the positive result the compute-bound
Phase-2 regime never produced.

## Success criteria

- A clean `min(N*, C_kv(context))` overlay with measured points and a labeled regime handoff, **or** a
  clean "prefill-bound throughout" result with SM-saturation evidence.
- If KV-bound is reached: ≥1 policy beating `nocap` on urgent-class SLO at that operating point, with
  ±std over ≥3 repeats.

## Cost / risk

Heavy, thermally-gated runs (24k prefills are slow + hot on the passive T4). Full sweep ≈ 1–2 h.
Gated behind the calibration probe so we don't burn the budget on a mis-tuned campaign.
