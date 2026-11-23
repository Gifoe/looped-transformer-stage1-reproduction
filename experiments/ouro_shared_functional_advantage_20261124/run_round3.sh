#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
if [[ ${1:-} == --wait ]]; then
    for ((check=0;check<2400;check++)); do
        if grep -qx DONE run_development.log; then break; fi
        sleep 5
    done
fi
grep -qx DONE run_development.log
CUDA_VISIBLE_DEVICES="" "$py" prepare_round3.py --results results
initial=$("$py" -c 'import json; print(json.load(open("results/ROUND3_LOCK.json"))["initial_file"])')
common=(--archive "$archive" --previous "$previous" --model "$model" --data data)
if [[ ! -s results/ON_POLICY_FIT/RECEIPT.json ]]; then
    bash control_budget.sh generate_training_prefixes on_policy.py "${common[@]}" --controller "$gen/results/FINAL_CONTROLLER.pt" \
        --out results/ON_POLICY_FIT > on_policy_prefixes.log 2>&1
fi
for arm in teacher_resume on_policy_resume; do
    opt=()
    if [[ "$arm" == on_policy_resume ]]; then opt=(--on-policy results/ON_POLICY_FIT/TRAIN_PREFIXES.jsonl); fi
    if [[ ! -s "results/fits/$arm/TRAIN.json" ]]; then
        bash control_budget.sh "train_$arm" train_advantage.py "${common[@]}" --moments results/MOMENT_FIT \
            --objective worst_advantage --initial "$initial" --steps 336 --lr .0005 --out "results/fits/$arm" \
            "${opt[@]}" > "train_$arm.log" 2>&1
    fi
    bash control_budget.sh "dev_$arm" evaluate_controller.py "${common[@]}" --arm "$arm" \
        --checkpoint "results/fits/$arm/controller.pt" --depths 2 3 4 --limit 64 \
        --out results/development > "dev_$arm.log" 2>&1
done
echo DONE
