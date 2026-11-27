#!/usr/bin/env bash
set -euo pipefail
v=/root/autodl-tmp/ouro-v5-icml-phase1-20261127
source /root/autodl-tmp/ouro-v4-cfit-20261126/handoff_codex/env.sh
cd "$v"
export PYTHONPATH="$v/codex_supervision:$v:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 7>codex_supervision/track_b_official_one_owner.lock
flock -w 10 7
while ! grep -Fxq 'V5_1C_GPU1_TRAINING_AND_DEV_DONE' logs/codex_v5_track_b_gpu1.log; do sleep 30; done
busy() {
    local uuid
    uuid=$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader)
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$uuid"
}
while busy 1; do sleep 15; done
export CUDA_VISIBLE_DEVICES=""
"$py" -u codex_supervision/v5_peft_official.py --root "$v" --phase seal
read -r method0 method1 < <("$py" -c 'import json; a=list(json.load(open("SELECTION_PHASE1_TRACK_B.json"))["candidates"]); assert len(a)==2; print(*a)')
for name in "$method0" "$method1"; do
    "$py" -u codex_supervision/v5_peft_official.py --root "$v" --phase freeze --alias "$name"
done
one() (
    set -e
    local gpu=$1 name=$2
    export CUDA_VISIBLE_DEVICES=$gpu
    for task in mmlu_pro mmlu hellaswag arc_easy piqa arc_challenge winogrande sciq; do
        "$py" -u codex_supervision/v5_peft_official.py --root "$v" --phase evaluate --alias "$name" --task "$task"
    done
)
if ! busy 0; then
    echo "V5_OFFICIAL_TWO_CARDS $(date --iso-8601=seconds)"
    one 0 "$method0" >"logs/codex_v5_official_card0.log" 2>&1 &
    p0=$!
    one 1 "$method1" >"logs/codex_v5_official_card1.log" 2>&1 &
    p1=$!
    rc=0
    wait "$p0" || rc=1
    wait "$p1" || rc=1
    if test "$rc" -ne 0; then echo V5_OFFICIAL_WORKER_ERROR; exit "$rc"; fi
else
    echo "V5_OFFICIAL_SINGLE_CARD_GPU0_ALREADY_OWNED $(date --iso-8601=seconds)"
    one 1 "$method0"
    one 1 "$method1"
fi
"$py" -u codex_supervision/v5_peft_official.py --root "$v" --phase aggregate
if grep -q '"status": "FAIL"' track_a_csbp/preflight/CSBP_PREFLIGHT_VERDICT.json; then
    "$py" -u codex_supervision/v5_phase1_close.py --root "$v"
fi

