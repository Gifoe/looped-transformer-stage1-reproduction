#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
protocol=results/FINAL_PROTOCOL.json
if [[ ${1:-} == --wait ]]; then
    for ((i=0;i<3600;i++)); do
        grep -qx DONE run_round4_and_freeze.log && break
        sleep 5
    done
    grep -qx DONE run_round4_and_freeze.log
fi
sha256sum -c --quiet FINAL_PROTOCOL.sha256
arms=$("$py" -c 'import json; print(" ".join(json.load(open("results/FINAL_PROTOCOL.json"))["allowed_run_order"]))')
stages=$("$py" -c 'import json; print(" ".join(json.load(open("results/FINAL_PROTOCOL.json"))["run_stages"]))')
for stage in $stages; do
    for arm in $arms; do
        bash control_budget.sh "test_${stage}_$arm" evaluate_suite.py --archive "$archive" --previous "$previous" --model "$model" \
            --protocol "$protocol" --arm "$arm" --stage "$stage" --out results/confirmation >> "confirmation_${arm}.log" 2>&1
    done
done
echo DONE
