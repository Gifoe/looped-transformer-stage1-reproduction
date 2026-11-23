#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
until test -s results/dev/DEV_PROPOSAL.json && [[ $(tail -1 GPU_LEDGER.tsv | cut -f1) == END ]]; do
    if [[ $(tail -1 GPU_LEDGER.tsv | cut -f1) == END ]] && [[ $(tail -1 GPU_LEDGER.tsv | cut -f6) != 0 ]]; then
        tail -12 schedule_dev.log >&2
        exit 1
    fi
    sleep 5
done
common=(--archive "$archive" --previous "$previous" --model "$model" --corpus "$corpus" --control "$control")
bash control_budget.sh fit_rank1 fit_rank1.py "${common[@]}" --out results/fit > fit_rank1.log 2>&1
bash control_budget.sh rescreen screen.py dev "${common[@]}" --ob-validation data/ob_validation.parquet \
    --new-weight results/fit/rank1_generation.pt --out results/dev > rescreen.log 2>&1
printf 'DONE\n'
