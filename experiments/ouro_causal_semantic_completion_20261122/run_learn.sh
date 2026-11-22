#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-causal-semantic-completion-20261122
source env.sh
until test "$(tail -1 oracle.log)" = DONE; do
    if ! ps -p "${1:?oracle PID}" -o stat= | grep -q -v Z; then echo "oracle terminated"; exit 1; fi
    sleep 10
done
sha256sum LEARNING_PROTOCOL.md csc_learn.py > results/LEARNING_CODE_LOCK.sha256
if "$py" -c 'import json,sys; sys.exit(not json.load(open("results/ORACLE_SUMMARY.json"))["gate"]["full"])'; then
    bash gpu_budget.sh learned csc_learn.py "$model" results
else
    printf 'Oracle gate did not pass; learning not started.\n'
fi
printf 'DONE\n'
