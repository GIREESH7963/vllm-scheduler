# LLM Inference Scheduling — Project Memory

Adaptive admission scheduling for multi-tenant LLM inference on vLLM, with a queueing model
and speculative-decoding workloads. This is a **research project**; the goal is a clean,
measured, *explained* characterization — not a production system.

## How this repo is wired for Claude Code

Work happens **one phase at a time**. Each phase has its own `CLAUDE.md` inside its folder.
Claude Code loads a subdirectory's `CLAUDE.md` **on demand** — the first time it reads a file
in that folder — so only the current phase's instructions enter context.

- This root file loads every session (it sits above the working dir). Keep it lean.
- Phase files load only when you work in that phase's folder. That is intentional.
- The phase links below are for **navigation**; a markdown link does NOT auto-load a file.
  To start a phase, `cd` into its folder and read its `CLAUDE.md` first.
- Do **not** `@import` the phase files here — imports load at launch and would put every phase
  in context at once, defeating the point.

## Phases (do them in order)

0. [phase0-setup](phase0-setup/CLAUDE.md) — vLLM + DCGM + first measurement · status: DONE
1. [phase1-harness](phase1-harness/CLAUDE.md) — workload generator + measurement harness · DONE
2. [phase2-policies](phase2-policies/CLAUDE.md) — admission/ordering scheduler + policy comparison · DONE
3. [phase3-model](phase3-model/CLAUDE.md) — queueing model + validation · DONE
4. [phase4-writeup](phase4-writeup/CLAUDE.md) — report, plots, repo polish · TODO

Update the status tag as you finish each phase.

## Repo layout

- `common/` — shared package: config loading, telemetry samplers, metrics aggregation, plotting
- `configs/` — one YAML per experiment (no hardcoded params in scripts)
- `phase0-setup/` … `phase4-writeup/` — per-phase scripts + that phase's `CLAUDE.md`
- `results/` — run outputs as JSON (git-ignored except committed summaries)

## Stack & commands (exact strings)

- Python 3.11, `uv` for envs. Serving: vLLM. GPU telemetry: DCGM + NVML.
- Install: `uv pip install -r requirements.txt`
- Start server (n-gram spec decode; `--dtype float16` is required on the T4 — no hardware bf16):
  `vllm serve <model> --dtype float16 --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}' --port 8000`
- Run an experiment: `python -m phaseN.run --config configs/<name>.yaml`
- Tests: `pytest -q`
- Format + lint: `ruff format . && ruff check .`

## Global conventions

- **Metrics schema** — every run writes one JSON to `results/<timestamp>_<config>.json` with:
  `config, throughput_tok_s, ttft_ms{p50,p99}, tpot_ms{p50,p99}, kv_occupancy, sm_active_pct,
  dram_active_pct, energy_j_per_tok, slo_attainment, acceptance_rate`.
- One config = one YAML in `configs/`. No magic numbers in scripts.
- Telemetry sources: vLLM `/metrics` is the primary live signal; DCGM for SM-active / DRAM-active;
  NVML for power only.
- Git: commit after each working step, small messages. Never commit `results/*.json` (only summaries).

## Global rules

- **IMPORTANT: schedule at the ADMISSION/ordering layer in FRONT of vLLM.** Do NOT modify vLLM's
  internal continuous-batching scheduler by default (that is a phase-2-only stretch, isolated).
- **IMPORTANT: always report p50 AND p99, never just the mean.**
- **IMPORTANT: warm up before measuring** — discard the first ~30s / N requests of every run.
- **IMPORTANT: spec-decode uses off-the-shelf methods (n-gram, EAGLE-3). We do NOT train drafts here.**
- `nvidia-smi` "GPU-util %" is NOT SM occupancy — use DCGM for utilization, NVML for power.
- **Hardware: local NVIDIA T4 (16GB).** It IS a data-center GPU, so DCGM profiling fields work (run
  `dcgmi profile -l` to confirm). `nv-hostengine` needs root — fine on the workstation. Linux/WSL2 only.
- **IMPORTANT (T4): always pass `--dtype float16`** — Turing has no hardware bf16.
- **T4 limits:** keep models to 1–3B (the 8B EAGLE-3 case does NOT fit in 16GB — use n-gram). Simulate
  multi-tenancy as multiple request streams on ONE vLLM instance, not several instances. Watch GPU temp
  on long runs (passively cooled) — thermal throttling adds measurement noise.
- Keep the base model small (1–3B) for fast iteration unless a config says otherwise.
- **Scope guard:** if a task grows past the current phase's "definition of done", STOP and record it
  as future work instead of expanding scope.
