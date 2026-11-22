#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
while [[ $(tail -n 1 full.log) != DONE ]]; do
    ps -p "${1:?full PID}" -o stat= | grep -qv Z
    sleep 8
done
sha256sum diagnostics.py > results/DIAGNOSTIC_CODE_LOCK.sha256
bash gpu_budget.sh diagnostic diagnostics.py --model "$model" --archive "$archive" --corpus "$corpus" \
    --prior "$previous" --out results/feedback
printf 'DONE\n'
