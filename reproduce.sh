#!/bin/bash
# End-to-end reproduction of the headline results and figures.
#
# Regenerates: Phase-1 harness sweeps -> Phase-2 policy matrices + predictor experiment ->
# Phase-3 queueing model + figures. Each run writes one JSON per run to results/, plus the
# committed summaries (results/summaries/*) and figures (results/figures/*).
#
# REQUIRES: the GPU box (NVIDIA T4, Linux/WSL2) with a vLLM server already running (step 0 below).
# Runs are THERMALLY GATED — each waits for the T4 to cool below the config's cooldown_c before
# measuring — so the full matrix is slow but safe on the passively-cooled T4 (budget ~2-3 h).
# Phase 3 alone is OFFLINE (no GPU/server) and re-fits the model from the committed CSV.
set -euo pipefail
cd "$(dirname "$0")"

# ---- 0. Environment + server (run these once, by hand, before this script) -------------------
#   uv venv --python 3.11 && source .venv/bin/activate
#   uv pip install -r requirements.txt
#   vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype float16 \
#     --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}' \
#     --port 8000
PY="${PY:-.venv/bin/python}"
POLICIES="nocap,fcfs,srpt,edf,adaptive"

echo "===== REPRODUCE START $(date -u +%FT%TZ) ====="

# ---- 1. Phase 1: measurement-harness sweeps (no-scheduler baseline) --------------------------
echo "----- [1/4] Phase-1 harness sweeps $(date -u +%TZ) -----"
"$PY" -u harness/run.py --config configs/phase1_validation.yaml   # fast fixed-workload check
"$PY" -u harness/run.py --config configs/phase1_mixed.yaml        # no-scheduler baseline sweep

# ---- 2. Phase 2: policy comparison matrices --------------------------------------------------
echo "----- [2/4] Phase-2 policy matrices $(date -u +%TZ) -----"
"$PY" -u scheduler/run.py --config configs/phase2_mixed.yaml --policies "$POLICIES" --results-dir results
"$PY" -u scheduler/run.py --config configs/phase2_chat.yaml  --policies "$POLICIES" --results-dir results

# ---- 3. Phase 2: output-length prediction mini-experiment ------------------------------------
echo "----- [3/4] Phase-2 predictor experiment $(date -u +%TZ) -----"
"$PY" -u scheduler/run.py --config configs/phase2_mixed.yaml --experiment predictor --results-dir results

# ---- 4. Phase 3: queueing model + figures (OFFLINE — no GPU needed) ---------------------------
# Re-fits M/G/1-PS + capacity model from the extracted measurements and regenerates Phase-3 figures.
echo "----- [4/4] Phase-3 model fit $(date -u +%TZ) -----"
"$PY" -u model/model.py --config configs/phase3_model.yaml

echo "===== REPRODUCE DONE $(date -u +%FT%TZ) ====="
echo "Summaries: results/summaries/   Figures: results/figures/   Report: report/report.md"
