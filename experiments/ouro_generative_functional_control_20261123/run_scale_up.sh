#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
for ((i=0;i<1200;i++)); do
    grep -qx DONE run_development.log && break
    sleep 3
done
grep -qx DONE run_development.log
args=(--archive "$archive" --previous "$previous" --model "$model")
dev() {
    local name=$1 domain=$2 mode=$3 n=$4
    shift 4
    local item="${name}_${domain}"
    local arm=()
    if [[ "$name" != native ]]; then arm=(--checkpoint "results/$name/controller.pt"); fi
    bash control_budget.sh "dev_$item" evaluate_generation.py "${args[@]}" "${arm[@]}" --data "data/dev_$domain.jsonl" \
        --dataset "$domain" --arm "$name" --limit "$n" --depths 4 --batch 32 --out results/development \
        --modes "$mode" "$@" >> development.log 2>&1
}
if [[ ! -s results/multitask_full/TRAIN.json ]]; then
    CUDA_VISIBLE_DEVICES="" "$py" rescore_development.py
    work=$("$py" -c 'import json; print(json.load(open("results/gen_rank/TRAIN.json"))["padded_token_loops"])')
    if [[ ! -e data/mixed_reasoning_train.jsonl ]]; then
        "$py" -c 'from task_data import load,save; x=load("data/mixed_train.jsonl"); save("data/mixed_reasoning_train.jsonl",[r for r in x if r["dataset"] in {"csqa","obqa","squad","gsm8k"}])'
    fi
    if [[ ! -s results/multitask_reasoning_small/TRAIN.json ]]; then
        bash control_budget.sh multitask_reasoning_small train_generative.py "${args[@]}" --data data/mixed_reasoning_train.jsonl \
            --rank-weight .25 --steps 3000 --max-work "$work" --out results/multitask_reasoning_small > multitask_reasoning_small.log 2>&1
    fi
    name=multitask_reasoning_small
    dev "$name" csqa content 160 letter candidate
    dev "$name" obqa content 96 letter candidate
    dev "$name" squad span 72
    dev "$name" gsm8k gsm_final 72
    for name in native gen_rank multitask_small multitask_reasoning_small; do
        dev "$name" no_robots other 24
        dev "$name" smol-constraints other 24
    done
    CUDA_VISIBLE_DEVICES="" "$py" rescore_development.py
    for ((i=0;i<300;i++)); do
        test -s data/SCALE_LOCK.json && break
        sleep 3
    done
    test -s data/SCALE_LOCK.json
    steps=$("$py" -c 'import json; d=json.load(open("data/SCALE_LOCK.json")); print((d["n"]+3)//4)')
    bash control_budget.sh multitask_full train_generative.py "${args[@]}" --data data/full_mixed_train.jsonl \
        --rank-weight .25 --steps "$steps" --out results/multitask_full > multitask_full.log 2>&1
fi
dev multitask_full csqa content 160 letter candidate
dev multitask_full obqa content 96 letter candidate
dev multitask_full squad span 72
dev multitask_full gsm8k gsm_final 72
dev multitask_full no_robots other 24
dev multitask_full smol-constraints other 24
CUDA_VISIBLE_DEVICES="" "$py" rescore_development.py
echo DONE
