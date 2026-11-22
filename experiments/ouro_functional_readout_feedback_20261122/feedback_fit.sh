#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
while [[ $(tail -n 1 dual.log) != DONE ]]; do
    ps -p "${1:?dual PID}" -o stat= | grep -qv Z
    sleep 6
done
mkdir -p results/feedback
sha256sum feedback.py feedback_paths.py > results/FEEDBACK_CODE_LOCK.sha256
opts=(--model "$model" --archive "$archive" --corpus "$corpus" --prior "$previous" --out results/feedback)
if [[ ! -s results/feedback/FEEDBACK_FIT.json ]]; then
    bash gpu_budget.sh feedback_fit feedback.py fit "${opts[@]}"
fi
if [[ ! -s results/feedback/DISTILL_FIT.json ]]; then
    bash gpu_budget.sh distill_fit feedback.py distill-fit "${opts[@]}"
fi
mapfile -t selected < <("$py" -c 'import json; c=json.load(open("results/READOUT_FREEZE.json"))["config"]["dual"]; print(c["file"]); print(c["dose"])')
bash gpu_budget.sh feedback_validation feedback.py validation "${opts[@]}" --readout "results/${selected[0]}" --readout-dose "${selected[1]}"
printf 'DONE\n'
