#!/usr/bin/env bash
set -euo pipefail
v=/root/autodl-tmp/ouro-v5-icml-phase1-20261127
source /root/autodl-tmp/ouro-v4-cfit-20261126/handoff_codex/env.sh
cd "$v"
export PYTHONPATH="$v/codex_supervision:$v:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1
exec 7>codex_supervision/track_b_official_one_owner.lock
flock -n 7 || exit 0
while ! grep -Fxq 'V5_1C_GPU1_TRAINING_AND_DEV_DONE' logs/codex_v5_track_b_gpu1.log; do sleep 40; done
gpu=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$gpu"; do sleep 15; done
"$py" -u codex_supervision/v5_peft_official.py --root "$v" --phase seal
methods=$("$py" -c 'import json; print(*json.load(open("SELECTION_PHASE1_TRACK_B.json"))["candidates"])')
for name in $methods; do
    "$py" -u codex_supervision/v5_peft_official.py --root "$v" --phase freeze --alias "$name"
done
for task in mmlu_pro mmlu hellaswag arc_easy piqa arc_challenge winogrande sciq; do
    for name in $methods; do
        "$py" -u codex_supervision/v5_peft_official.py --root "$v" --phase evaluate --alias "$name" --task "$task"
    done
done
"$py" -u codex_supervision/v5_peft_official.py --root "$v" --phase aggregate
if grep -q '"status": "FAIL"' track_a_csbp/preflight/CSBP_PREFLIGHT_VERDICT.json; then
    "$py" -u codex_supervision/v5_phase1_close.py --root "$v"
fi

