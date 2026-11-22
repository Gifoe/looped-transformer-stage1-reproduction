#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-causal-semantic-completion-20261122
source env.sh
until test "$(tail -1 bank.log)" = DONE; do
    if ! ps -p "${1:?bank PID}" -o stat= | grep -q -v Z; then echo "bank terminated"; exit 1; fi
    sleep 10
done
sha256sum PROTOCOL.md csc_oracle.py > results/ORACLE_CODE_LOCK.sha256
bash gpu_budget.sh oracle csc_oracle.py "$model" results
printf 'DONE\n'
