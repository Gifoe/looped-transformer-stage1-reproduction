#!/usr/bin/env bash
set -euo pipefail
source env.sh
until [[ $(tail -n 1 confirm.log) == DONE ]]; do
    ps -p "$1" -o stat= | grep -qv Z
    sleep 13
done
if [[ ! -s results/SUBSPACE_ANALYSIS/SUBSPACE_RANDOM_INIT_CONTROLS.json ]]; then
    bash gpu_budget.sh subspace_random_init subspace_init_control.py --model "$model" --archive "$archive" --previous "$previous" --corpus "$corpus" --out results > random_init.log 2>&1
fi
"$py" generation_compare.py results
"$py" subspace_transfer.py results/SUBSPACE_ANALYSIS --external
"$py" report.py results REPORT_ZH.md
printf "DONE\n"
