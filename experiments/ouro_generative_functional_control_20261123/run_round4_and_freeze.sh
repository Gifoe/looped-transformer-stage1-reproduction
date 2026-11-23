#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
for ((i=0;i<3000;i++)); do
    grep -qx DONE run_scale_up.log && test -s data/filtered/CONTAMINATION.json && break
    sleep 4
done
grep -qx DONE run_scale_up.log
test -s data/filtered/CONTAMINATION.json
if [[ ! -e results/ROUND4_DECISION.json ]]; then
    CUDA_VISIBLE_DEVICES="" "$py" prepare_selection.py
fi
args=(--archive "$archive" --previous "$previous" --model "$model")
extra=()
arm_args=()
if "$py" -c 'import json,sys; sys.exit(not json.load(open("results/ROUND4_DECISION.json"))["run_paired_retention"])'; then
    initial=$("$py" -c 'import json; print(json.load(open("results/ROUND4_DECISION.json"))["shared_initial"])')
    for arm in retention_control retention_strong; do
        strength=.25
        if [[ "$arm" == retention_strong ]]; then strength=.90; fi
        if [[ ! -s "results/$arm/TRAIN.json" ]]; then
            bash control_budget.sh "$arm" train_generative.py "${args[@]}" --data data/retention_train.jsonl \
                --initial "results/$initial/controller.pt" --steps 282 --rank-weight "$strength" --lr .0004 \
                --out "results/$arm" > "$arm.log" 2>&1
        fi
        for group in csqa obqa squad gsm8k; do
            case "$group" in
                csqa) opts=(content letter candidate); n=160 ;;
                obqa) opts=(content letter candidate); n=96 ;;
                squad) opts=(span); n=72 ;;
                gsm8k) opts=(gsm_final); n=72 ;;
            esac
            bash control_budget.sh "dev_${arm}_${group}" evaluate_generation.py "${args[@]}" --checkpoint "results/$arm/controller.pt" \
                --data "data/dev_$group.jsonl" --dataset "$group" --modes "${opts[@]}" --depths 4 --batch 32 \
                --arm "$arm" --limit "$n" --out results/development >> development.log 2>&1
        done
        extra+=(--extra "$arm")
    done
    CUDA_VISIBLE_DEVICES="" "$py" rescore_development.py
fi
if [[ ! -s results/FINAL_PROTOCOL.json ]]; then
    historical="$prior_transfer/results/HOLDOUT/csqa/source/s261123137/k234/controller.pt"
    CUDA_VISIBLE_DEVICES="" "$py" freeze_test.py --result results --data data --filtered data/filtered \
        --historical "$historical" --out results/FINAL_PROTOCOL.json "${extra[@]}"
    sha256sum results/FINAL_PROTOCOL.json > FINAL_PROTOCOL.sha256
fi
sha256sum -c --quiet FINAL_PROTOCOL.sha256
echo DONE
