# Phase 2 — Admission/ordering scheduler + policy comparison (Days 15–28)

Root: [../CLAUDE.md](../CLAUDE.md) · Prev: [phase1-harness](../phase1-harness/CLAUDE.md) · Next: [phase3-model](../phase3-model/CLAUDE.md)

## Goal
Put a scheduling layer in FRONT of vLLM that decides admission order (and, for multi-tenancy,
routing across engine instances), and compare policies on the phase-1 workloads.

## Definition of done
- A pluggable scheduler interface with: FCFS, token-aware SJF/SRPT (size = prompt-length proxy),
  EDF (deadlines), and one adaptive feedback policy.
- An output-length predictor mini-experiment measuring the cost of misprediction on SRPT.
- A head-to-head comparison across ≥2 workload mixes producing the core result plots.

## Tasks
1. `phase2-policies/scheduler.py`: queue + policy interface; each policy a small strategy class.
2. FCFS baseline; SRPT-proxy using prompt length; EDF using per-request deadlines.
3. Adaptive feedback policy (rule-based): widen admission when `sm_active` is low; deprioritize
   long jobs when `kv_occupancy` > threshold; throttle admission when p99 TTFT rises.
4. Output-length predictor: a light heuristic/regressor; compare SRPT-with-oracle vs
   SRPT-with-prediction vs prompt-length-proxy.
5. Run the matrix (policies × mixes × arrival rates); write results + plots.

## Rules
- **IMPORTANT: this is an ADMISSION/ordering layer. Do NOT modify vLLM internals.** You control
  which requests enter and in what order, NOT intra-engine batch formation. State this in code docs.
- **IMPORTANT (adaptive ≠ ML):** the adaptive policy is a tuned feedback rule. No learned/RL policy
  in scope — that is future work.
- Multi-tenancy on the 16GB T4 = multiple request **streams** (with different priorities/deadlines)
  hitting ONE vLLM instance. Don't run several full instances — you'll exceed 16GB VRAM.

## Stretch (only if ahead; keep isolated)
- Make ONE in-engine knob adaptive (batch-size cap or KV-admission under pressure). This is the
  ONLY place vLLM-internal changes are allowed, and only after the core comparison is done.

## When done
Commit, flip status, move to phase3.
