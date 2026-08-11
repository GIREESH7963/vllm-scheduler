#!/usr/bin/env bash
# Experiment F — re-run C and E at a cap that does not destroy the measurement.
#
# Experiments C and E both ran at max_num_seqs=1024 and both measured nothing. The reason is the
# same for each, and it is not a bug in either experiment: `max_num_seqs` is not a scheduler cap.
# vLLM sizes its memory-profiling pass at the cap, reserves the resulting activation peak, and
# gives KV the remainder —
#
#     cap  256 -> activation 2.52 GiB, KV 4.74 GiB, "Maximum concurrency" 4.22x
#     cap 1024 -> activation 5.64 GiB, KV 1.63 GiB, "Maximum concurrency" 1.44x
#
# — so raising the cap to give the engine room to reach N=256 did the opposite. Both C arms and
# both E arms pinned at ~99% KV occupancy near N ~ 125-149 and never died: C could not tell the
# scorer from the cap, and E's MiB/seq column came out NaN in all six trials.
#
# F fixes this in the order that puts the decisive result first:
#
#   1. calibration  — measure the cap/KV trade-off directly (~50 min, no workload). Produces the
#                     figure for the regime result and picks C's cap from data.
#   2. F-2  k=7 at the default cap — THE decisive arm, and it landed: 3/3 deaths at 4.640,
#                     4.645 and 4.642 MiB/seq against 4.637 predicted (+0.1%), with the ratio to
#                     the k=4 reference 1.602 observed vs 1.600 predicted (8/5). Cap 256, KV
#                     ceiling at N ~ 363. It measures the law's (k+1) term and breaks C's
#                     scorer/cap confound in one run.
#
#                     The confound breaks on KV OCCUPANCY at death, not concurrency at death:
#                     the three trials died at 43.6%, 43.6% and 63.3% occupancy, leaving a third
#                     or more of the KV pool free, so neither the cap nor exhaustion accounts for
#                     them. Concurrency does not carry the argument on its own — the deaths came
#                     at N = 186, 187 and 256, and that last one is the cap itself.
#
#                     An earlier draft of this header read "Boundary at N ~ 129 ... a death at
#                     129 can only be the scorer". That was a units error: 129 came off the rate
#                     column of docs/experiment_b.md, which is a request rate in req/s, not a
#                     concurrency. Do not reintroduce it.
#   3. F-1  k=2 at the default cap — predicted null; the arm that can falsify the law cheaply.
#   4. F-3  C re-run at the calibrated cap, if one qualifies. Corroboration: the k=7 arm already
#                     separates scorer from cap, so this is skipped without loss if the window
#                     turns out to be empty.
#
# k=7 runs before k=2 deliberately. If the campaign is interrupted, the arm that carries the
# result is already in.
#
# Detached, per the way this box is used — launch and disconnect:
#   setsid nohup ./phase3-limits/run_expF.sh > /dev/null 2>&1 < /dev/null &
#   ./status.sh
#   tail -f results/expF/expF_run.log
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="results/expF"
mkdir -p "$OUT"
PY=.venv/bin/python
LOG="$OUT/expF_run.log"

log() { echo "[expF $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---- do not race another campaign --------------------------------------------------------------
if pgrep -f "run_expCD.sh|run_expE.sh|oom_probe.py|phase2-policies/run.py|vllm serve" >/dev/null 2>&1; then
    log "ANOTHER CAMPAIGN IS RUNNING — refusing to start. Check ./status.sh"
    exit 1
fi

wait_for_cool() {
    for _ in $(seq 1 120); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0)
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
        if [ "${t:-99}" -le 77 ] && [ "${u:-9999}" -lt 1000 ]; then return 0; fi
        sleep 10
    done
    log "WARNING: gave up waiting for cooldown (temp=${t:-?}C used=${u:-?}MiB)"
}

run_arm() {   # $1 = tag, $2 = config, $3 = trials
    wait_for_cool
    log "arm [$1]: $2 (${3} trials)"
    $PY -u phase3-limits/oom_probe.py --config "$2" --trials "$3" \
        --out-dir "$OUT/$1" 2>&1 | sed "s/^/  [F:$1] /" | tee -a "$LOG"
    log "arm [$1] done (rc=${PIPESTATUS[0]})"
}

log "EXPERIMENT F START"

# ---- 1. cap calibration -------------------------------------------------------------------------
log "step 1/4: cap calibration"
$PY -u tools/calibrate_cap.py --out "$OUT/calibration.json" 2>&1 \
    | sed 's/^/  [F:calib] /' | tee -a "$LOG"
log "calibration done (rc=${PIPESTATUS[0]})"

# ---- 2. the decisive arm ------------------------------------------------------------------------
log "step 2/4: k=7 at the default cap — the decisive arm"
run_arm k7_cap256 configs/scorer_k7_qwen3b_cap256.yaml 3

# ---- 3. the predicted null ----------------------------------------------------------------------
log "step 3/4: k=2 at the default cap — predicted null"
run_arm k2_cap256 configs/scorer_k2_qwen3b_cap256.yaml 3

# ---- 4. experiment C, if the calibration found a usable cap -------------------------------------
log "step 4/4: experiment C re-run"
if $PY -u tools/make_regime_configs.py --calibration "$OUT/calibration.json" 2>&1 \
        | sed 's/^/  [F:cfg] /' | tee -a "$LOG"; then
    CAP=$(cat configs/.regime_cap 2>/dev/null || echo "")
    if [ -n "$CAP" ]; then
        run_arm regime_spec_on_cap"$CAP"  configs/regime_qwen3b_spec_on_cap"$CAP".yaml 3
        run_arm regime_spec_off_cap"$CAP" configs/regime_qwen3b_spec_off_cap"$CAP".yaml 3
    else
        log "no cap written; skipping C"
    fi
else
    log "SKIPPING C: no cap above 256 leaves enough KV to reach the scorer boundary."
    log "  This is a result, not a failure — it means the cap and the scorer cannot be separated"
    log "  by raising the cap on this GPU, and the k=7 arm is the evidence that separates them."
fi

# ---- analysis -----------------------------------------------------------------------------------
log "analysing"
$PY -u tools/analyze_expF.py --results-dir "$OUT" --out "$OUT/expF_analysis.json" 2>&1 \
    | sed 's/^/  /' | tee -a "$LOG"

log "EXPERIMENT F COMPLETE"
log "read next: $OUT/expF_analysis.json, $OUT/calibration.json, then fold into docs/experiment_b.md"
