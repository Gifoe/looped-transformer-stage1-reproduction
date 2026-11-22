#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-causal-semantic-completion-20261122
source env.sh
until test "$(tail -n 1 lookup.log)" = DONE; do
    if ! ps -p "${1:?lookup PID}" -o stat= | grep -q -v Z; then echo "lookup terminated"; exit 1; fi
    sleep 10
done
sha256sum csc_diagnostics.py > results/DIAGNOSTICS_CODE_LOCK.sha256
if test -f results/LEARNED_SUMMARY_K1.json && test -f results/LEARNED_SUMMARY_K2.json; then
    bash gpu_budget.sh diagnostics csc_diagnostics.py "$model" results
else
    printf 'Learning did not run; operator ablations not started.\n'
fi
printf 'DONE\n'
