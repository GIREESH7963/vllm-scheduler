#!/bin/bash
# KV-regime experiment launcher (see report/kv_regime_experiment.md).
# Traces the compute->KV crossover by sweeping prompt context, then a policy comparison in whatever
# regime is reached. Thermally gated; run from repo root with the vLLM server already up and PINNED
# (see step 0 in the design doc — record num_gpu_blocks / KV budget from the server startup log).
#
# CALIBRATION GATE: run step 1 alone first and inspect kv_occupancy / num_waiting / num_running in the
# resulting JSON before uncommenting the full sweep. Do not burn the thermal budget on a mis-tuned run.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"

echo "===== KV-REGIME START $(date -u +%FT%TZ) ====="

# ---- 1. Calibration probe (do this first, alone) ---------------------------------------------
echo "----- calibration probe: 8k context $(date -u +%TZ) -----"
"$PY" -u harness/run.py --config configs/kvsweep_ctx8k.yaml --results-dir results

# ---- 2. Context sweep (uncomment after the probe looks right) --------------------------------
# echo "----- context sweep 2k -> 24k $(date -u +%TZ) -----"
# "$PY" -u harness/run.py --config configs/kvsweep_ctx2k.yaml  --results-dir results
# "$PY" -u harness/run.py --config configs/kvsweep_ctx16k.yaml --results-dir results
# "$PY" -u harness/run.py --config configs/kvsweep_ctx24k.yaml --results-dir results

# ---- 3. Policy comparison in the KV-bound regime (only if step 2 reached a KV-bound point) ----
# echo "----- KV-bound policy comparison $(date -u +%TZ) -----"
# "$PY" -u scheduler/run.py --config configs/kvbound_policies.yaml \
#   --policies nocap,fcfs,srpt,edf,adaptive --results-dir results

echo "===== KV-REGIME DONE $(date -u +%FT%TZ) ====="
echo "Next: extract per-run JSONs to model/kvsweep_runs.csv, then re-run model/model.py to draw"
echo "results/figures/phase3_kv_regime_validation.png"
