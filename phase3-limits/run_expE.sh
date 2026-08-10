#!/usr/bin/env bash
# Experiment E — validate the scorer memory law beyond its N term.
#
#     M_scorer = N * (k+1) * V * 4 bytes
#
# Every death observed so far used k=4 on a model with V=151936, so only N has been varied. The
# law is currently a coefficient fitted on one axis and asserted on three. E varies the other
# two:
#
#   E-1  k=2  -> 1.739 MiB/seq, predicts death near N ~ 344
#   E-2  k=7  -> 4.637 MiB/seq, predicts death near N ~ 129
#   E-3  V=128256 (Llama-3.2-1B) -> predicts 656.7 MiB at N=256, against Qwen's 741.9
#
# The measured quantity is MiB per sequence: reported failed allocation divided by concurrency
# at the failing step. It must come out at 1.739 / 2.898 / 4.637 for k = 2 / 4 / 7 across three
# well-separated concurrencies, and at 2.898 * (128256/151936) = 2.446 for the Llama arm.
#
# E-3 is gated behind a HuggingFace licence and is skipped with a logged message if the model is
# unavailable, so the k arms complete unattended either way.
#
#   ./phase3-limits/run_expE.sh
#   tail -f results/expE/expE_run.log
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="results/expE"
mkdir -p "$OUT"
PY=.venv/bin/python

log() { echo "[expE $(date -u +%H:%M:%S)] $*"; }

# ---- wait for the C/D chain ------------------------------------------------------------------
log "waiting for experiments C and D to finish"
while pgrep -f "run_expCD.sh" >/dev/null 2>&1; do sleep 60; done
# The C/D chain ends by regenerating figures, which is CPU-only but slow; make sure the GPU is
# genuinely idle and nothing else is mid-teardown before claiming it.
while pgrep -f "oom_probe.py|phase2-policies/run.py|vllm serve" >/dev/null 2>&1; do sleep 30; done
log "C and D finished"

wait_for_cool() {
    for _ in $(seq 1 120); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0)
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
        if [ "${t:-99}" -le 77 ] && [ "${u:-9999}" -lt 1000 ]; then return 0; fi
        sleep 10
    done
    log "WARNING: gave up waiting for cooldown (temp=${t:-?}C used=${u:-?}MiB)"
}

run_arm() {   # $1 = tag, $2 = config
    wait_for_cool
    log "Experiment E [$1]: $2"
    $PY -u phase3-limits/oom_probe.py --config "$2" --trials 3 \
        --out-dir "$OUT/$1" 2>&1 | sed "s/^/  [E:$1] /"
    log "Experiment E [$1] done (rc=${PIPESTATUS[0]})"
}

# ---- E-1 / E-2: vary k ------------------------------------------------------------------------
run_arm k2 configs/scorer_k2_qwen3b.yaml
run_arm k7 configs/scorer_k7_qwen3b.yaml

# ---- E-3: vary V, if the gated model is reachable ---------------------------------------------
LLAMA=meta-llama/Llama-3.2-1B-Instruct
# Check file access, not repo metadata: model_info() succeeds on a gated repo without a token
# and would send us into three doomed 10-minute server-startup timeouts.
if $PY - <<PYCHK >/dev/null 2>&1
import sys
from huggingface_hub import hf_hub_download
try:
    hf_hub_download("$LLAMA", "config.json")
except Exception:
    sys.exit(1)
PYCHK
then
    log "Llama-3.2-1B reachable — running the V arm"
    run_arm vocab_llama1b configs/scorer_vocab_llama1b.yaml
else
    log "SKIPPING E-3: $LLAMA is gated and no usable HF token was found."
    log "  To enable: accept the licence at https://huggingface.co/$LLAMA,"
    log "  export HF_TOKEN=hf_..., then re-run ./phase3-limits/run_expE.sh"
fi

# ---- report -----------------------------------------------------------------------------------
log "EXPERIMENT E COMPLETE"
$PY - <<'PYSUM' 2>&1 | sed 's/^/  /'
import glob, json
from pathlib import Path
V = 151936
print(f"{'arm':<16} {'trial':<6} {'died':<5} {'alloc MiB':>10} {'N':>6} {'MiB/seq':>8} {'predicted':>10}")
for arm, k1 in (("k2", 3), ("k7", 8), ("vocab_llama1b", 5)):
    v = 128256 if arm.startswith("vocab") else V
    pred = k1 * v * 4 / 2**20
    for f in sorted(glob.glob(f"results/expE/{arm}/*trial*.json")):
        d = json.loads(Path(f).read_text())
        o = d.get("oom") or {}
        peaks = d.get("peaks") or {}
        n = peaks.get("num_running")
        a = o.get("failed_alloc_mib")
        ratio = (a / n) if (a and n) else float("nan")
        print(f"{arm:<16} {d.get('trial'):<6} {str(d.get('died')):<5} "
              f"{(a if a else float('nan')):>10.0f} {(n if n else float('nan')):>6.0f} "
              f"{ratio:>8.3f} {pred:>10.3f}")
print("\nThe MiB/seq column is the test: it must match the prediction, which is the only")
print("quantity in the law that (k+1) and V control. N is free to differ between arms.")
PYSUM
log "read next: results/expE/, then fold into docs/experiment_b.md 5"
