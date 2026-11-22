#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-causal-semantic-completion-20261122
source env.sh
until test "$(tail -n 1 learned.log)" = DONE; do
    if ! ps -p "${1:?learned PID}" -o stat= | grep -q -v Z; then echo "learning terminated"; exit 1; fi
    sleep 10
done
sha256sum csc_lookup.py > results/LOOKUP_CODE_LOCK.sha256
if test -f results/LEARNED_SUMMARY_K1.json && test -f results/LEARNED_SUMMARY_K2.json; then
    bash gpu_budget.sh lookup csc_lookup.py "$model" results
else
    printf 'Learning did not run; lookup transfer not started.\n'
fi
printf 'DONE\n'
