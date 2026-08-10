#!/usr/bin/env bash
# Progress check for the detached experiment runs.
#
#   ./status.sh          one-shot snapshot
#   ./status.sh -w       refresh every 30 s (Ctrl-C to leave; does not affect the runs)
#
# Safe to run any time, from any SSH session. Read-only.
cd "$(dirname "$0")"

show() {
  echo "=============================================================================="
  echo " vllm_scheduler status   $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "=============================================================================="

  echo
  echo "-- GPU ----------------------------------------------------------------------"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
             --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"

  echo
  echo "-- Running jobs -------------------------------------------------------------"
  local any=0
  if pgrep -f "run_expB.sh" >/dev/null 2>&1; then
    echo "  [RUNNING] Experiment B  (model-size comparison)"; any=1
  fi
  if pgrep -f "oom_probe.py" >/dev/null 2>&1; then
    echo "  [RUNNING] OOM probe"; any=1
  fi
  if pgrep -f "phase1-harness/run.py" >/dev/null 2>&1; then
    echo "  [RUNNING] sweep harness"; any=1
  fi
  pgrep -af "vllm serve" 2>/dev/null | grep -oE 'serve [^ ]+' | sed 's/^/  serving: /' | head -2
  [ "$any" -eq 0 ] && echo "  (no experiment processes running)"

  echo
  echo "-- Experiment A: OOM reproduction -------------------------------------------"
  if [ -f results/expA/expA_summary_corrected.json ]; then
    .venv/bin/python - <<'PY' 2>/dev/null
import json
try:
    d = json.load(open('results/expA/expA_summary_corrected.json'))
except Exception:
    raise SystemExit
n = sum(1 for r in d if (r.get('oom_corrected') or {}).get('failed_alloc_mib'))
print(f"  COMPLETE: OOM reproduced in {n}/{len(d)} independent trials")
for r in d:
    o = r.get('oom_corrected') or {}
    a = r.get('at_failure_corrected') or {}
    kv = a.get('peak_kv_occupancy')
    print(f"    trial {r['trial']}: {o.get('failed_alloc_mib','?')} MiB at "
          f"{str(o.get('alloc_site'))[:34]}, N={a.get('peak_num_running')}, "
          f"KV={kv*100:.1f}%" if kv else f"    trial {r['trial']}")
PY
  else
    echo "  not finished"
  fi

  echo
  echo "-- Experiment B: 1.5B vs 3B -------------------------------------------------"
  for tag in 1.5b 3b; do
    c=$(ls results/*expB_qwen${tag}_r*.json 2>/dev/null | wc -l)
    printf "  sweep %-5s : %2d / 21 runs\n" "$tag" "$c"
  done
  c=$(ls results/expB/oom3b/*trial*.json 2>/dev/null | wc -l)
  echo "  3B OOM probe: $c / 3 trials"
  if grep -q "EXPERIMENT B COMPLETE" results/expB/expB_run.log 2>/dev/null; then
    echo "  >> EXPERIMENT B COMPLETE"
  fi
  echo "  last log lines:"
  grep -E "^\[expB" results/expB/expB_run.log 2>/dev/null | tail -3 | sed 's/^/    /'

  echo
  echo "-- Artifacts ----------------------------------------------------------------"
  printf "  paper-v1 snapshot : %s\n" "$([ -d snapshots/paper-v1 ] && echo 'frozen (read-only)' || echo 'MISSING')"
  printf "  statistics report : %s\n" "$([ -f results/stats/STATISTICS.md ] && echo 'results/stats/STATISTICS.md' || echo 'pending')"
  printf "  queueing model    : %s\n" "$([ -f docs/queueing_model.md ] && echo 'docs/queueing_model.md' || echo 'pending')"
  printf "  expB analysis     : %s\n" "$([ -f docs/experiment_b.md ] && echo 'docs/experiment_b.md' || echo 'pending')"
  printf "  paper figures     : %s PDFs in results/figures/paper\n" "$(ls results/figures/paper/*.pdf 2>/dev/null | wc -l)"
  printf "  git HEAD          : %s\n" "$(git log --oneline -1 2>/dev/null || echo 'n/a')"
  echo
}

if [ "${1:-}" = "-w" ]; then
  while true; do clear; show; sleep 30; done
else
  show
fi
