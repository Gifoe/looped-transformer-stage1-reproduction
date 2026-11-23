#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
for ((i=0;i<600;i++)); do
    if grep -qx DONE run_confirmation.log; then break; fi
    sleep 3
done
grep -qx DONE run_confirmation.log
sha256sum --check --quiet FORECAST_LOCK.sha256
for domain in arc csqa; do
    bash control_budget.sh "CONTEXT_$domain" context_counterfactual.py --archive "$archive" --previous "$previous" \
        --model "$model" --rank1 "$t1" --domain "$domain" --data "data/${domain}_confirm.jsonl" \
        --forecast-lock results/FORECAST_LOCK.json --out "results/CONTEXT_$domain" > "CONTEXT_${domain}.log" 2>&1
    echo "DONE CONTEXT_$domain"
done
echo DONE
