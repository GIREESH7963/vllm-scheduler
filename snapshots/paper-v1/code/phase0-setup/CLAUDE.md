# Phase 0 — Setup & baseline (Days 1–4)

Root: [../CLAUDE.md](../CLAUDE.md) · Next: [phase1-harness](../phase1-harness/CLAUDE.md)

## Goal
Get vLLM serving with n-gram speculative decoding, wire the telemetry pipeline, and produce
one reproducible single-stream measurement end to end.

## Definition of done
- `vllm serve` runs a small model with n-gram spec decode on a data-center GPU.
- A smoke script sends a few requests and writes one valid results JSON (schema in root).
- Telemetry captured for that run: vLLM `/metrics` + DCGM SM-active/DRAM-active + NVML power.
- `results/` has one baseline file; a summary of it is committed.

## Tasks
1. Env (Linux or WSL2): `uv pip install -r requirements.txt` (vllm, pynvml, httpx, pyyaml, pandas, matplotlib).
2. Install DCGM; start `nv-hostengine` (needs root — you have it). Confirm the T4 exposes profiling
   fields: `dcgmi profile -l -i 0` (should list sm_active, dram_active, tensor_active, …). Then stream
   `dcgmi dmon -e 1002,1004,1005` (SM-active / tensor-active / DRAM-active); `dcgmi dmon -l` lists all IDs.
3. Start the server on the T4 with **`--dtype float16`** and the n-gram config from root; confirm
   `/metrics` is reachable. (bf16 will fail on Turing.)
4. `common/telemetry.py`: samplers for vLLM `/metrics`, DCGM (via `dcgmi` or dcgm-exporter),
   and NVML power (pynvml). Sample on a fixed interval during a run.
5. `phase0-setup/smoke.py`: fire ~20 requests, collect telemetry, write one results JSON.
6. Sanity-check: TTFT and tok/s reasonable, SM-active non-zero, `acceptance_rate` present.

## Gotchas (local T4)
- The T4 supports DCGM profiling (it's a data-center GPU) — but `nv-hostengine` must run with root.
- **Always `--dtype float16`** — Turing has no hardware bf16.
- Take **power from NVML** (or DCGM field 155), not from a "prof" field.
- DCGM profiling conflicts with Nsight — don't run both at once.
- The T4 is passively cooled: watch GPU temp on longer runs; thermal throttling adds noise.
- Give the server a warmup burst before trusting any numbers.
- `num_speculative_tokens + 1` must stay within vLLM's per-step token limit.

## When done
Commit, flip the phase-0 status in root to DONE, move to phase1.
