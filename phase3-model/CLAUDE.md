# Phase 3 — Queueing model + validation (Days 29–38)

Root: [../CLAUDE.md](../CLAUDE.md) · Prev: [phase2-policies](../phase2-policies/CLAUDE.md) · Next: [phase4-writeup](../phase4-writeup/CLAUDE.md)

## Goal
Model the system with queueing theory, then show where the model predicts behavior and where
KV-cache capacity forces it to break down. **This is the project's differentiator — protect it.**

## Definition of done
- An analytical model: continuous batching ≈ M/G/1 **processor sharing**; token-aware ≈ SRPT;
  extended with a HARD KV-cache **capacity limit**.
- Predicted latency/throughput-vs-load curves overlaid on the phase-1/2 measurements.
- A written explanation of the deviation (queueing/blocking once KV capacity saturates).

## Tasks
1. `phase3-model/model.py`: implement the M/G/1-PS + capacity model; feed it measured
   service-time and acceptance-rate statistics from `results/`.
2. Overlay predicted vs measured; identify the load region where they diverge.
3. Explain WHY: the capacity cap → admission blocking; batch can't grow past the KV limit.
4. Figures: model-vs-measurement per workload mix.

## Rules
- **IMPORTANT:** be able to derive the crossover from first principles — write the reasoning in
  the report, not just plots.
- Keep the model simple and legible. A clear approximate model that *explains* the gap beats a
  complex one that fits only by tuning.

## When done
Commit, flip status, move to phase4.
