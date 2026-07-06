# Phase 0 — Baseline (committed summary)

**Run:** `20260706T084014Z_phase0_qwen1.5b.json` · **Date:** 2026-07-06
**Model:** Qwen/Qwen2.5-1.5B-Instruct · **GPU:** NVIDIA T4 (16 GB, driver 580.159) · dtype float16
**Serving:** vLLM 0.8.5.post1, n-gram spec decode `{num_speculative_tokens:4, prompt_lookup_min:2, prompt_lookup_max:5}`
**Workload:** single stream (concurrency 1), 8 warmup + 20 measured requests, 128 max tokens, greedy.

| Metric | Value |
|---|---|
| throughput_tok_s | 59.7 |
| TTFT ms (p50 / p99) | 55.3 / 88.9 |
| TPOT ms (p50 / p99) | 16.2 / 16.4 |
| kv_occupancy | 0.0005 (single stream, short seqs) |
| sm_active_pct (DCGM) | 71.0 |
| dram_active_pct (DCGM) | 67.0 |
| mean_power_w (NVML) | 67.6 |
| energy_j_per_tok | 1.13 |
| slo_attainment (TTFT≤500ms, TPOT≤50ms) | 1.00 |
| acceptance_rate (n-gram draft) | 0.458 |

**Cross-check:** vLLM server log independently reports "Draft acceptance rate: 0.458"
(counters 484 accepted / 1056 draft), matching the `/metrics`-derived value — telemetry validated.

## What this establishes (Definition of Done)
- ✅ vLLM serves a 1–3B model with n-gram speculative decoding on the T4 (`--dtype float16`).
- ✅ Telemetry pipeline wired: vLLM `/metrics` (KV occ, queue depth, spec counters) + DCGM
  `dmon` (SM-active/DRAM-active) + NVML power — all three sources produce non-zero data.
- ✅ `phase0-setup/smoke.py` fires warmup+measured bursts and writes one schema-valid results JSON.
- ✅ Warmup discarded before measuring; p50 AND p99 reported; power from NVML, utilization from DCGM.

## Notes / gotchas found
- vLLM 0.8.5 resolver pulls `transformers 5.x` which breaks the Qwen2 tokenizer; **pinned
  `transformers==4.51.3`** in `requirements.txt`.
- Spec-decode acceptance counters (`vllm:spec_decode_num_{accepted,draft}_tokens_total`) only
  appear in `/metrics` once drafting is active — a short probe request won't surface them.
- Baseline is deliberately single-stream. Concurrency/batching/GPU saturation is Phase-1+ scope.
- Environment is not a git repo yet — this summary is the committed artifact; run `git init` when ready.
