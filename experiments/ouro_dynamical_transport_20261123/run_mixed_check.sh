#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
for ((i=0;i<600;i++)); do
    if grep -qx DONE run_context.log; then break; fi
    sleep 3
done
grep -qx DONE run_context.log
for domain in arc csqa; do
    bash control_budget.sh "E2_MIXED_$domain" e2_signed.py --archive "$archive" --previous "$previous" \
        --model "$model" --rank1 "$transport_archive/results/HOLDOUT/mixed/source/s261123137/k234/controller.pt" \
        --n 48 --cell 2:4 --cell 4:6 --data "$PWD/data/${domain}_confirm.jsonl" \
        --out "results/E2_MIXED_$domain" > "E2_MIXED_$domain.log" 2>&1
    echo "DONE E2_MIXED_$domain"
done
echo DONE
