#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
axis="$prior_root/results/CAUSAL_ANALYSIS/mean_direction_rank1.pt"
until [[ -s results/FREEZE.json ]]; do
  ps -p "$1" -o stat= | grep -qv Z
  sleep 9
done
while ps -p "$1" -o stat= | grep -qv Z; do sleep 4; done
printf "RUN D\n"
bash control_budget.sh D_finite_dose experiment_d.py --archive "$archive" --model "$model" --corpus "$corpus" --direction "$axis" --out results/D_FINITE_DOSE > dose.log 2>&1
printf "COMPLETE D\nRUN CONFIRM\n"
bash control_budget.sh confirm confirm_control.py --archive "$archive" --previous "$previous" --model "$model" --corpus "$corpus" --external "$external" --prior "$prior_root" --out results > confirm_control.log 2>&1
printf "COMPLETE CONFIRM\nDONE\n"
