#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
while [[ $(tail -n 1 direction_audit.log) != DONE ]]; do
    ps -p "${1:?audit PID}" -o stat= | grep -qv Z
    sleep 6
done
test "$(tail -n 1 feedback_fit.log)" = DONE
test -s results/CALIBRATION_FREEZE.json
opts=(--model "$model" --archive "$archive" --corpus "$corpus" --prior "$previous" --out results/feedback)
if [[ ! -e results/CONFIRMATION_LOCK.sha256 ]]; then
    sha256sum readout_eval.py calibration.py feedback.py feedback_paths.py results/READOUT_FREEZE.json \
              results/CALIBRATION_FREEZE.json results/feedback/FREEZE.json external/openbookqa.parquet > results/CONFIRMATION_LOCK.sha256
fi
sha256sum -c results/CONFIRMATION_LOCK.sha256
complete() {
    [[ -s "$1" ]] && "$py" -c 'import json,sys; sys.exit(len(json.load(open(sys.argv[1]))["cost"])!=int(sys.argv[2]))' "$1" "$2"
}
if [[ ! -s results/feedback/MECHANISM.json ]]; then
    bash gpu_budget.sh feedback_mechanism feedback.py mechanism "${opts[@]}" > feedback_mechanism.log 2>&1
fi
if ! complete results/feedback/TEST_COST.json 4; then
    bash gpu_budget.sh feedback_arc feedback.py test "${opts[@]}" > feedback_test.log 2>&1
fi
if [[ ! -s results/ARC_READOUT_COST.json || ! -s results/OPENBOOKQA_READOUT_COST.json ]]; then
    bash readout_test.sh > readout_test.log 2>&1
fi
if ! complete results/feedback/EXTERNAL_COST.json 4; then
    bash gpu_budget.sh feedback_external feedback.py external "${opts[@]}" --external-data external/openbookqa.parquet > feedback_external.log 2>&1
fi
if ! complete results/feedback/TEST_CONTROL.json 16; then
    bash gpu_budget.sh feedback_controls feedback.py controls "${opts[@]}" > feedback_controls.log 2>&1
fi
printf 'DONE\n'
