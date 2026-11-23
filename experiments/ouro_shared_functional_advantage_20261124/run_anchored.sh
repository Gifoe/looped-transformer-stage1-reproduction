#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
if [[ ${1:-} == --wait ]]; then
    for ((check=0;check<2400;check++)); do
        if grep -qx DONE run_round3.log; then break; fi
        sleep 5
    done
fi
grep -qx DONE run_round3.log
base=results/fits/mixed/controller.pt
gpu_args=(--archive "$archive" --previous "$previous" --model "$model" --data data)
bash control_budget.sh train_anchored_policy train_advantage.py "${gpu_args[@]}" --moments results/MOMENT_FIT \
    --objective worst_advantage --initial "$base" --reference "$base" --steps 336 --lr .00035 \
    --on-policy results/ON_POLICY_FIT/TRAIN_PREFIXES.jsonl --out results/fits/anchored_policy > train_anchored_policy.log 2>&1
bash control_budget.sh dev_anchored_policy evaluate_controller.py "${gpu_args[@]}" --arm anchored_policy \
    --checkpoint results/fits/anchored_policy/controller.pt --depths 2 3 4 --limit 64 \
    --out results/development > dev_anchored_policy.log 2>&1
echo DONE
