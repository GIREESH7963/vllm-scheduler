# Phase 1 — Workload generator + measurement harness (Days 5–14)

Root: [../CLAUDE.md](../CLAUDE.md) · Prev: [phase0-setup](../phase0-setup/CLAUDE.md) · Next: [phase2-policies](../phase2-policies/CLAUDE.md)

## Goal
Drive vLLM with realistic, reproducible multi-tenant load and log every metric cleanly.
This harness is the foundation every later phase reuses — get it solid.

## Definition of done
- Generator produces load with a **Poisson arrival process** and configurable prompt/output-length
  distributions and workload mixes, including a **spec-decode class**.
- Async load driver hits the server at a target arrival rate and records per-request timings.
- Harness writes one results JSON per run (root schema) with correct p50/p99 and warmup handling.
- A plotting helper turns a set of runs into latency / throughput / utilization figures.

## Tasks
1. `common/workload.py`: request model with prompt-length + target output-length draws
   (short / long / reasoning-like / coding-like profiles) and a per-stream `spec_decode` flag.
2. `common/loadgen.py`: async client (httpx) issuing requests per Poisson(λ); capture arrival,
   TTFT, per-token times, completion. Support N concurrent tenants.
3. `common/metrics.py`: aggregate to the root schema; **discard warmup**; compute p50/p99.
4. `common/plots.py`: latency-vs-load, throughput-vs-load, sm_active-vs-load.
5. `configs/`: define 3–4 workload mixes (one spec-decode-heavy).
6. Validate: a fixed workload gives stable, comparable reports across repeated runs.

## Rules
- **IMPORTANT: measurement rigor is the whole point** — warmup, fixed seeds, ≥3 repeats,
  report variance, always p50/p99.
- Arrival process is **open-loop (Poisson)**, not closed-loop, unless a config says otherwise.

## Gotchas
- Output length is unknown ahead of time — record **actual vs requested** length; phase-2 needs this.
- Don't let the client be the bottleneck; verify it isn't CPU-bound at high λ.

## When done
Commit, flip status, move to phase2.
