#!/usr/bin/env bash
# Experiment A — OOM reproduction. Launched detached so it survives an SSH disconnect.
#
#   ./phase3-limits/run_expA.sh [trials]
#
# Progress:  tail -f results/expA/expA_run.log
# Stop:      kill -TERM -"$(cat results/expA/expA.pid)"     # negative PID = whole process group
set -euo pipefail
cd "$(dirname "$0")/.."

TRIALS="${1:-3}"
OUT="results/expA"
mkdir -p "$OUT/logs"

# setsid detaches from the controlling terminal, so closing the laptop / dropping the SSH
# session cannot deliver SIGHUP to the run or to the vLLM servers it spawns.
setsid nohup .venv/bin/python phase3-limits/oom_probe.py \
    --config configs/oomprobe_qwen1.5b.yaml \
    --trials "$TRIALS" \
    --out-dir "$OUT" \
    > "$OUT/expA_run.log" 2>&1 &

echo $! > "$OUT/expA.pid"
echo "Experiment A started detached: PID $(cat "$OUT/expA.pid"), $TRIALS trials"
echo "  log: $OUT/expA_run.log"
