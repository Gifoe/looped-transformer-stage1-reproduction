#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-causal-semantic-completion-20261122
source env.sh
until test "$(tail -n 1 diagnostics.log)" = DONE; do
    if ! ps -p "${1:?diagnostics PID}" -o stat= | grep -q -v Z; then echo "diagnostics terminated"; exit 1; fi
    sleep 10
done
sha256sum csc_template.py > results/TEMPLATE_CODE_LOCK.sha256
if test -f results/LEARNED_SUMMARY_K1.json && "$py" -c 'import json,sys; sys.exit(json.load(open("results/LEARNED_SUMMARY_K1.json"))["gate"]["passes"])'; then
    bash gpu_budget.sh template csc_template.py "$model" results
else
    printf 'Rank-one follow-up did not meet its launch condition.\n'
fi
printf 'DONE\n'
