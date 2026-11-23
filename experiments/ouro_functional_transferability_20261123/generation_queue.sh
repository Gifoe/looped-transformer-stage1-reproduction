#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
for ((i=0; i<720; i++)); do
    if grep -qx DONE gamma_confirm.log; then break; fi
    sleep 3
done
grep -qx DONE gamma_confirm.log
common=(--archive "$archive" --previous "$previous" --model "$model" --minimal "$prior_root" \
    --results results --data-manifest data/optional/ROUND2_MANIFEST.json --out results/TRANSFER_GENERATION)
CUDA_VISIBLE_DEVICES="" "$py" transfer_generation.py "${common[@]}" --verify-only
bash control_budget.sh r2_greedy transfer_generation.py "${common[@]}"
printf 'DONE\n'
