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
  c=$(ls results/expB/oom3b/*trial*.json results/expB/oom3b_more/*trial*.json 2>/dev/null | wc -l)
  echo "  3B OOM probe: $c / 10 trials (3 initial + 7 follow-up for reproduction rate)"
  if grep -q "EXPERIMENT B COMPLETE" results/expB/expB_run.log 2>/dev/null; then
    echo "  >> EXPERIMENT B COMPLETE"
  fi
  echo "  last log lines:"
  grep -E "^\[expB" results/expB/expB_run.log 2>/dev/null | tail -3 | sed 's/^/    /'

  echo
  echo "-- Experiment C: regime dependence (spec on/off, max_num_seqs 1024) ----------"
  for tag in spec_on spec_off; do
    c=$(ls results/expCD/regime_${tag}/*trial*.json 2>/dev/null | wc -l)
    printf "  %-8s : %d / 3 trials\n" "$tag" "$c"
  done
  # Both arms completed and neither died, which looks like a clean null but is not one: the
  # cap of 1024 shrank KV to 1.63 GiB and the engine went KV-bound before reaching the scorer.
  # Say so here, because "3 / 3 trials" on its own reads as a result.
  echo "  NOTE: superseded — cap=1024 shrank KV to 1.63 GiB, both arms went KV-bound near"
  echo "        N~125 without dying, so this never tested scorer-vs-cap. See experiment F."

  echo
  echo "-- Experiment D: powered policy comparison (n=10) ----------------------------"
  for tag in mixed_n10 chat_n10; do
    c=$(ls results/*phase2_${tag}_*.json 2>/dev/null | wc -l)
    printf "  %-10s : %2d / 50 runs\n" "$tag" "$c"
  done
  if grep -q "EXPERIMENTS C AND D COMPLETE" results/expCD/expCD_run.log 2>/dev/null; then
    echo "  >> C AND D COMPLETE"
  fi
  grep -E "^\[expCD" results/expCD/expCD_run.log 2>/dev/null | tail -2 | sed 's/^/    /'

  echo
  echo "-- Experiment F: C and E re-run at a cap that preserves KV --------------------"
  if [ -f results/expF/calibration.json ]; then
    echo "  cap calibration : done"
    .venv/bin/python - <<'PYCAL' 2>/dev/null
import json
d = json.load(open("results/expF/calibration.json"))
for p in d["points"]:
    if p.get("error"):
        print(f"    cap {p['max_num_seqs']:>4} k={p['num_speculative_tokens']}: {p['error']}")
    else:
        print(f"    cap {p['max_num_seqs']:>4} k={p['num_speculative_tokens']}: "
              f"activation {p['activation']:.2f} GiB, KV {p['kv']:.2f} GiB, "
              f"est KV-bound N~{p['est_kv_bound_n']}")
r = d.get("recommendation", {})
print(f"    C re-run cap: {r.get('cap') or 'none qualifies — ' + r.get('note', '')[:60]}")
PYCAL
  else
    echo "  cap calibration : running or pending"
  fi
  for tag in k7_cap256 k2_cap256; do
    c=$(ls results/expF/${tag}/*trial*.json 2>/dev/null | wc -l)
    d=$(grep -l '"died": true' results/expF/${tag}/*trial*.json 2>/dev/null | wc -l)
    printf "  %-10s : %d / 3 trials, %d died\n" "$tag" "$c" "$d"
  done
  for dir in results/expF/regime_spec_*/; do
    [ -d "$dir" ] || continue
    c=$(ls "$dir"/*trial*.json 2>/dev/null | wc -l)
    printf "  %-10s : %d / 3 trials\n" "$(basename "$dir")" "$c"
  done
  if grep -q "EXPERIMENT F COMPLETE" results/expF/expF_run.log 2>/dev/null; then
    echo "  >> EXPERIMENT F COMPLETE — read results/expF/expF_analysis.json"
  fi
  grep -E "^\[expF" results/expF/expF_run.log 2>/dev/null | tail -3 | sed 's/^/    /'

  echo
  echo "-- Artifacts ----------------------------------------------------------------"
  for s in snapshots/*/; do
    [ -d "$s" ] || continue
    n=$(basename "$s")
    if [ -w "$s" ]; then state='WRITABLE — not frozen'; else state='frozen (read-only)'; fi
    printf "  %-17s : %s\n" "$n snapshot" "$state"
  done
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
