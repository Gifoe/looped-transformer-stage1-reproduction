#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
for ((i=0;i<600;i++)); do
    if grep -qx DONE run_negative_path.log; then break; fi
    sleep 3
done
grep -qx DONE run_negative_path.log
sha256sum --check --quiet PATH_LOCK.sha256
for name in arc csqa qasc; do
    domain=$name
    weight=$t1
    if [[ "$name" == qasc ]]; then
        domain=arc
        weight="$transport_archive/results/HOLDOUT/qasc/source/s261123137/k234/controller.pt"
    fi
    bash control_budget.sh "PATH_$name" path_integration.py --archive "$archive" --previous "$previous" --model "$model" \
        --rank1 "$weight" --lock results/PATH_LOCK.json --tag "$name" --data "data/path/${domain}_confirm.jsonl" \
        --out "results/PATH_$name" > "PATH_${name}.log" 2>&1
    echo "DONE PATH_$name"
done
echo DONE
