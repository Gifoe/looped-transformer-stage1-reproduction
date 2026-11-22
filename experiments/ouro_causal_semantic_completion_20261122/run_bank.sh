#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-causal-semantic-completion-20261122
source env.sh
until test "$(tail -1 native.log)" = DONE; do
    if ! ps -p "${1:?native PID}" -o stat= | grep -q -v Z; then echo "native terminated"; exit 1; fi
    sleep 8
done
sha256sum csc_bank.py > results/BANK_LOCK.sha256
bash gpu_budget.sh bank csc_bank.py "$model" results
printf 'DONE\n'
