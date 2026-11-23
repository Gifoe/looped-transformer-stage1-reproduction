#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
grep -qx DONE run_fitting.log
rank="$prior_transfer/results/HOLDOUT/csqa/source/s261123137/k234/controller.pt"
retention="$gen/results/FINAL_CONTROLLER.pt"
full="$gen/results/multitask_full/controller.pt"
common=(--archive "$archive" --previous "$previous" --model "$model" --data data --out results/development)

run() {
    local label=$1 weight=$2
    local args=()
    if [[ "$weight" != native ]]; then args=(--checkpoint "$weight"); fi
    bash control_budget.sh "dev_$label" evaluate_controller.py "${common[@]}" --arm "$label" \
        --depths 2 3 4 --limit 64 "${args[@]}" > "dev_${label}.log" 2>&1
}

run native native
run historical_rank "$rank"
run historical_retention "$retention"
run historical_mixed "$full"
for arm in ranking generation mixed linear_weak mean_advantage worst_advantage worst_rank_init worst_mixed_init; do
    run "$arm" "results/fits/$arm/controller.pt"
done
echo DONE
