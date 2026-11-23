#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
if [[ ${1:-} == --wait ]]; then
    until grep -qx DONE run_e2.log; do sleep 4; done
fi
for domain in arc csqa; do
    bash control_budget.sh "e3_find_$domain" e3_paths.py --archive "$archive" --previous "$previous" --model "$model" \
        --transport "$transport_archive" --rank1 "$t1" --data "$transport_archive/data/${domain}_discovery.jsonl" \
        --domain "$domain" --cell 1:4 --cell 2:4 --n 24 --batch 4 --out "results/E3_FIND_$domain" > "e3_find_${domain}.log" 2>&1
done
echo DONE
