#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-lookup-conflict-20261121
source env.sh
sha256sum PROTOCOL.md conf_test.py conf_stats.py smoke.py > results/CONF_LOCK.sha256
until test "$(tail -1 pairs.log)" = DONE; do
    if ! ps -p "${1:?paired PID}" -o stat= | grep -q -v Z; then echo "paired terminated"; exit 1; fi
    sleep 8
done
if "$py" -c 'import json,sys; sys.exit(not json.load(open("results/PAIR_SUMMARY.json"))["gate"])'; then
    "$py" -u smoke.py "$model" /root/ouro-lookup-onset-20261121/results/PAIRED.json
    "$py" -u conf_test.py "$model" results
fi
printf 'DONE\n'
