# Phase 4 — Write-up & polish (Days 39–45)

Root: [../CLAUDE.md](../CLAUDE.md) · Prev: [phase3-model](../phase3-model/CLAUDE.md)

## Goal
Turn the work into a clean, credible artifact: a short report, final figures, and a
reproducible repo.

## Definition of done
- 4–6 page report (or blog post): problem, admission-layer framing, policy comparison,
  queueing model + explained deviation, honest limitations, future work.
- Final figures regenerated from `results/` via `common/plots.py`.
- README: what it is, how to reproduce (env, server command, run command), results summary.
- One reproducibility script that regenerates the headline plots from raw results.

## Tasks
1. `phase4-writeup/report.md` (export to PDF if wanted).
2. Regenerate all figures; check they match the numbers cited in the text.
3. Write the README with exact reproduce steps.
4. `reproduce.sh`: env setup → representative runs → figures.
5. Future-work section: learned policy, GPU colocation, the in-engine knob.

## Rules
- **IMPORTANT: describe contributions accurately.** The `speculators` PRs contributed
  draft-training *loss functions*; this project serves OFF-THE-SHELF drafts. Do NOT claim you
  trained drafts.
- **IMPORTANT: frame honestly.** A rigorously-measured null result ("admission control adds little
  on top of vLLM's batcher") is a valid finding. Do NOT overclaim beating production systems.
- State the boundary plainly: single GPU, small models, admission layer.

## When done
Tag a release. Link the repo in your SOP and outreach to Gandhi.
