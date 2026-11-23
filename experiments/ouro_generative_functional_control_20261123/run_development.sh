#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
grep -qx DONE run_core_b.log
model_args=(--archive "$archive" --previous "$previous" --model "$model")
rank="$prior_transfer/results/HOLDOUT/csqa/source/s261123137/k234/controller.pt"

eval_development() {
    local method=$1 weight=$2
    local group opts n
    local weight_args=()
    if [[ "$weight" != native ]]; then weight_args=(--checkpoint "$weight"); fi
    for group in csqa obqa squad gsm8k; do
        case "$group" in
            csqa) opts=(content letter candidate); n=160 ;;
            obqa) opts=(content letter candidate); n=96 ;;
            squad) opts=(span); n=72 ;;
            gsm8k) opts=(gsm_final); n=72 ;;
        esac
        bash control_budget.sh "dev_${method}_${group}" evaluate_generation.py "${model_args[@]}" "${weight_args[@]}" \
            --data "data/dev_$group.jsonl" --dataset "$group" --modes "${opts[@]}" --depths 4 --batch 32 \
            --arm "$method" --limit "$n" --out results/development >> development.log 2>&1
    done
}
eval_development native native
eval_development rank_only "$rank"
eval_development gen_only results/gen_only/controller.pt
eval_development gen_rank results/gen_rank/controller.pt
for ((i=0;i<360;i++)); do
    if [[ -s data/DATA_LOCK.json ]]; then break; fi
    sleep 3
done
test -s data/DATA_LOCK.json
work=$("$py" -c 'import json; print(json.load(open("results/gen_rank/TRAIN.json"))["padded_token_loops"])')
if [[ ! -s results/multitask_small/TRAIN.json ]]; then
    bash control_budget.sh multitask_small train_generative.py "${model_args[@]}" --data data/mixed_train.jsonl \
        --rank-weight .25 --steps 3000 --max-work "$work" --out results/multitask_small > multitask_small.log 2>&1
fi
eval_development multitask_small results/multitask_small/controller.pt
echo DONE
