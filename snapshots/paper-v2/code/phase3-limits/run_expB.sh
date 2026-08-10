#!/usr/bin/env bash
# Experiment B — does the resource bottleneck move as model size changes?
#
# Both arms run under the SAME driver and the SAME harness. The 1.5B arm is re-measured rather
# than reused from paper-v1: the NVIDIA userspace driver was upgraded between that campaign and
# now (580.159.03 -> 580.173.02), so reusing the old numbers would confound "model size" with
# "driver version". Re-running costs ~30 min and removes the confound entirely.
#
# Stages per arm: baseline + concurrency sweep + KV occupancy + memory + throughput (the sweep
# harness records all of these per run), then the OOM boundary via the probe.
#
#   ./phase3-limits/run_expB.sh
#   tail -f results/expB/expB_run.log
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="results/expB"
mkdir -p "$OUT/logs"
PORT=8000
PY=.venv/bin/python
SPEC='{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'

log() { echo "[expB $(date -u +%H:%M:%S)] $*"; }

start_server() {  # $1 = model, $2 = logfile
    log "starting server for $1"
    setsid nohup .venv/bin/vllm serve "$1" \
        --port "$PORT" --dtype float16 --gpu-memory-utilization 0.9 \
        --speculative-config "$SPEC" > "$2" 2>&1 &
    echo $! > "$OUT/server.pid"
    for _ in $(seq 1 300); do
        if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            log "server healthy for $1"; return 0
        fi
        if ! kill -0 "$(cat "$OUT/server.pid")" 2>/dev/null; then
            log "SERVER DIED during startup for $1 — see $2"; return 1
        fi
        sleep 2
    done
    log "server for $1 did not become healthy in 600s"; return 1
}

stop_server() {
    [ -f "$OUT/server.pid" ] || return 0
    local pid; pid=$(cat "$OUT/server.pid")
    log "stopping server pid $pid"
    kill -TERM -"$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$OUT/server.pid"
    # Wait for the allocator to actually hand memory back before the next arm starts.
    for _ in $(seq 1 40); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
        [ "${used:-0}" -lt 1000 ] && break
        sleep 3
    done
    log "gpu free (used=${used:-?} MiB)"
}
trap stop_server EXIT

# ---- sweeps: baseline / concurrency / KV / memory / throughput, both model sizes ----
for ARM in "1.5b:Qwen/Qwen2.5-1.5B-Instruct" "3b:Qwen/Qwen2.5-3B-Instruct"; do
    TAG="${ARM%%:*}"; MODEL="${ARM#*:}"
    if start_server "$MODEL" "$OUT/logs/server_${TAG}.log"; then
        log "sweep: $MODEL"
        $PY -u phase1-harness/run.py --config "configs/expB_qwen${TAG}.yaml" \
            --results-dir results 2>&1 | sed "s/^/  [${TAG}] /" || log "sweep for $TAG failed"
    else
        log "SKIPPING sweep for $TAG (server would not start)"
    fi
    stop_server
done

# ---- OOM boundary for the larger model (1.5B boundary = Experiment A, same environment) ----
log "OOM boundary probe: Qwen2.5-3B"
$PY -u phase3-limits/oom_probe.py --config configs/oomprobe_qwen3b.yaml \
    --trials 3 --out-dir "$OUT/oom3b" 2>&1 | sed 's/^/  [oom3b] /' || log "3B OOM probe failed"

log "EXPERIMENT B COMPLETE"
