#!/bin/bash
# Phase-2 full data collection, chained so it runs unattended under one nohup.
# Thermally gated (each run waits for the T4 to cool to configs' cooldown_c before measuring),
# so this is slow-but-safe on the passively-cooled T4. Logs progress with timestamps.
set -u
cd ~/vllm_scheduler
PY=.venv/bin/python
POLICIES=nocap,fcfs,srpt,edf,adaptive

echo "===== PHASE 2 START $(date -u +%FT%TZ) ====="

echo "----- [1/3] mixed policy matrix $(date -u +%TZ) -----"
$PY -u phase2-policies/run.py --config configs/phase2_mixed.yaml --policies "$POLICIES" --results-dir results

echo "----- [2/3] predictor experiment $(date -u +%TZ) -----"
$PY -u phase2-policies/run.py --config configs/phase2_mixed.yaml --experiment predictor --results-dir results

echo "----- [3/3] chat policy matrix $(date -u +%TZ) -----"
$PY -u phase2-policies/run.py --config configs/phase2_chat.yaml --policies "$POLICIES" --results-dir results

echo "===== PHASE 2 DONE $(date -u +%FT%TZ) ====="
