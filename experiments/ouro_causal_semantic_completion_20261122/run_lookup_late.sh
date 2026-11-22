#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-causal-semantic-completion-20261122
source env.sh
until test "$(tail -n 1 formation.log)" = DONE; do
    if ! ps -p "${1:?formation PID}" -o stat= | grep -q -v Z; then echo "formation terminated"; exit 1; fi
    sleep 10
done
sha256sum csc_lookup_late.py > results/LOOKUP_LATE_CODE_LOCK.sha256
if test -f results/LOOKUP_SUMMARY.json; then
    bash gpu_budget.sh lookup_late csc_lookup_late.py "$model" results
else
    printf 'Lookup screen did not run.\n'
fi
printf 'DONE\n'
