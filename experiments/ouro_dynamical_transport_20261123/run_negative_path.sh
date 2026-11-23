#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
for ((i=0;i<600;i++)); do
    if grep -qx DONE run_mixed_check.log; then break; fi
    sleep 3
done
grep -qx DONE run_mixed_check.log
weights="$transport_archive/results/HOLDOUT/qasc/source/s261123137/k234/controller.pt"
bash control_budget.sh E3_QASC e3_paths.py --archive "$archive" --previous "$previous" --model "$model" \
    --transport "$transport_archive" --rank1 "$weights" --domain arc --data "$PWD/data/arc_confirm.jsonl" \
    --out results/E3_QASC --cell 2:4 --n 48 --batch 6 --forecast-lock results/FORECAST_LOCK.json > E3_QASC.log 2>&1
echo DONE
