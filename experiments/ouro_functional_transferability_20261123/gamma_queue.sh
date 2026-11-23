#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
for ((i=0; i<600; i++)); do
    if grep -qx DONE source_round2.log; then break; fi
    sleep 3
done
grep -qx DONE source_round2.log
bash control_budget.sh gamma_confirm gamma_intervention.py score --archive "$archive" --previous "$previous" --model "$model" \
    --results results --out results/GAMMA --data data/optional/arc_round2_confirm.jsonl \
    --data data/optional/csqa_round2_confirm.jsonl
printf 'DONE\n'
