#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-causal-semantic-completion-20261122
source env.sh
until test "$(tail -n 1 template.log)" = DONE; do
    if ! ps -p "${1:?template PID}" -o stat= | grep -q -v Z; then echo "template terminated"; exit 1; fi
    sleep 10
done
sha256sum csc_formation.py > results/FORMATION_CODE_LOCK.sha256
if "$py" -c 'import json,sys; sys.exit(not json.load(open("results/ORACLE_SUMMARY.json"))["gate"]["full"])'; then
    bash gpu_budget.sh formation csc_formation.py "$model" results
else
    printf 'Oracle gate did not pass; matched formation comparison not started.\n'
fi
printf 'DONE\n'
