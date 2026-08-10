#!/usr/bin/env bash
# Experiments C and D, chained to run unattended behind the in-flight 3B probe batch.
#
# C — Is the concurrency limit a property of the active resource regime?
#     Every death so far landed at N = 256, which is also the default max_num_seqs, so
#     "the scorer allocation ran out of room" and "the sequence cap did something" are
#     confounded. C raises the cap to 1024 and toggles speculative decoding:
#
#       C-a  spec ON,  cap 1024  -> predicts death near N ~ 256 anyway (scorer-bound),
#                                   with the cap never reached.
#       C-b  spec OFF, cap 1024  -> predicts no OOM at all; N climbs toward the KV limit
#                                   (~400 at the measured 0.2501%/seq) and plateaus under
#                                   preemption instead of crashing.
#
#     Together those exhibit the regime change that has so far only been extrapolated.
#
# D — Power for the admission-control claim. The n=3 campaign resolved nothing: 0 of 96
#     contrasts survived Holm correction. D re-runs the two cells where policies actually
#     separate (mixed λ=4, chat λ=6) at n=10, all five policies, on one server in one
#     session.
#
#   ./phase3-limits/run_expCD.sh
#   tail -f results/expCD/expCD_run.log
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="results/expCD"
mkdir -p "$OUT/logs"
PORT=8000
PY=.venv/bin/python
POLICIES=nocap,fcfs,srpt,edf,adaptive
SPEC='{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'

log() { echo "[expCD $(date -u +%H:%M:%S)] $*"; }

# ---- wait for the 3B probe batch already in flight -----------------------------------------
log "waiting for the running 3B probe batch to finish"
while pgrep -f "oom_probe.py" >/dev/null 2>&1; do sleep 30; done
log "3B probe batch finished ($(ls results/expB/oom3b_more/*trial*.json 2>/dev/null | wc -l) trials written)"

wait_for_cool() {   # the T4 is passively cooled; start every stage from a comparable state
    for _ in $(seq 1 120); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0)
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
        if [ "${t:-99}" -le 77 ] && [ "${u:-9999}" -lt 1000 ]; then return 0; fi
        sleep 10
    done
    log "WARNING: gave up waiting for cooldown (temp=${t:-?}C used=${u:-?}MiB)"
}

# ============================================================================================
# Experiment C — regime dependence
# ============================================================================================
for ARM in "spec_on:configs/regime_qwen3b_spec_on_cap1024.yaml" \
           "spec_off:configs/regime_qwen3b_spec_off_cap1024.yaml"; do
    TAG="${ARM%%:*}"; CFG="${ARM#*:}"
    wait_for_cool
    log "Experiment C [$TAG]: $CFG"
    $PY -u phase3-limits/oom_probe.py --config "$CFG" --trials 3 \
        --out-dir "$OUT/regime_$TAG" 2>&1 | sed "s/^/  [C:$TAG] /"
    log "Experiment C [$TAG] done (rc=${PIPESTATUS[0]})"
done

# ============================================================================================
# Experiment D — powered policy comparison
# ============================================================================================
start_server() {
    log "starting 1.5B server for phase-2 (spec decode on, default max_num_seqs)"
    setsid nohup .venv/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct \
        --port "$PORT" --dtype float16 --gpu-memory-utilization 0.9 \
        --speculative-config "$SPEC" > "$OUT/logs/server_phase2.log" 2>&1 &
    echo $! > "$OUT/server.pid"
    for _ in $(seq 1 300); do
        curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { log "server healthy"; return 0; }
        kill -0 "$(cat "$OUT/server.pid")" 2>/dev/null || { log "SERVER DIED during startup"; return 1; }
        sleep 2
    done
    log "server did not become healthy in 600s"; return 1
}

stop_server() {
    [ -f "$OUT/server.pid" ] || return 0
    local pid; pid=$(cat "$OUT/server.pid")
    log "stopping server pid $pid"
    kill -TERM -"$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$OUT/server.pid"
    for _ in $(seq 1 40); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
        [ "${used:-0}" -lt 1000 ] && break
        sleep 3
    done
    log "gpu free (used=${used:-?} MiB)"
}
trap stop_server EXIT

# Scoped to phase2_mixed λ=4 only. Each run measures for 33 s but the harness then waits for the
# passively-cooled T4 to fall back to 77 C, so the real cost is ~220 s per run — 85% of it
# cooling. Two cells would have been ~6 h. This cell carries by far the larger policy separation
# (nocap SLO 0.908 vs fcfs 0.122), so it is the one that can actually settle the claim.
wait_for_cool
if start_server; then
    CFG=configs/phase2_mixed_n10.yaml
    log "Experiment D: $CFG  (5 policies x 10 repeats, ~3 h)"
    $PY -u phase2-policies/run.py --config "$CFG" --policies "$POLICIES" \
        --results-dir results 2>&1 | sed "s/^/  [D] /"
    log "Experiment D done (rc=${PIPESTATUS[0]})"
else
    log "SKIPPING Experiment D (server would not start)"
fi
stop_server

# ============================================================================================
# Regenerate every derived artefact, so the analysis is finished on return rather than the
# raw data merely being present. Each step is independent; a failure in one is logged and the
# rest still run.
# ============================================================================================
log "regenerating derived artefacts"
$PY -u tools/fit_queueing_model.py --results-dir results 2>&1 | sed 's/^/  [fit] /' \
    || log "WARNING: queueing-model fit failed"
$PY -u tools/analyze_expB.py 2>&1 | sed 's/^/  [expB] /' \
    || log "WARNING: Experiment B analysis failed"
$PY -u tools/analyze_stats.py --results-dir results --baseline fcfs 2>&1 | sed 's/^/  [stats] /' \
    || log "WARNING: policy statistics failed"
$PY -u tools/make_paper_figures.py 2>&1 | sed 's/^/  [figs] /' \
    || log "WARNING: figure generation failed"

log "EXPERIMENTS C AND D COMPLETE"
log "summary:"
for tag in spec_on spec_off; do
    n=$(ls "$OUT/regime_$tag"/*trial*.json 2>/dev/null | wc -l)
    log "  C/$tag: $n trials"
done
log "  D: $(ls results/*phase2_mixed_n10_*.json 2>/dev/null | wc -l) / 50 runs"
log "  3B probe: $(ls results/expB/oom3b/*trial*.json results/expB/oom3b_more/*trial*.json 2>/dev/null | wc -l) / 10 trials"
log "read next: docs/experiment_b.md, results/stats/STATISTICS.md, results/figures/paper/"
