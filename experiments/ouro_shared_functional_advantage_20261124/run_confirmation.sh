#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
sha256sum -c --quiet CONFIRMATION.sha256
protocol=results/CONFIRMATION_LOCK.json

for stage in k4 transfer; do
    key=run_order
    if [[ "$stage" == transfer ]]; then key=transfer_arms; fi
    arms=$("$py" -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))[sys.argv[2]]))' "$protocol" "$key")
    for arm in $arms; do
        weight=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_path"][sys.argv[2]])' "$protocol" "$arm")
        opt=()
        if [[ "$weight" != native ]]; then opt=(--checkpoint "$weight"); fi
        bash control_budget.sh "test_${stage}_$arm" evaluate_controller.py --archive "$archive" --previous "$previous" \
            --model "$model" --data data/confirmation --protocol "$protocol" --out results/confirmation \
            --arm "$arm" --slice "$stage" "${opt[@]}" >> "test_$arm.log" 2>&1
    done
done
echo DONE
