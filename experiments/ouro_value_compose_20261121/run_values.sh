#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-value-compose-20261121
source env.sh
sha256sum PROTOCOL.md value_test.py value_stats.py > results/VALUE_LOCK.sha256
until test "$(tail -1 native.log)" = DONE; do
    if ! ps -p "${1:?native PID}" -o stat= | grep -q -v Z; then echo "native terminated"; exit 1; fi
    sleep 8
done
if "$py" -c 'import json,sys; sys.exit(not json.load(open("results/NATIVE_SUMMARY.json"))["gate"])'; then
    "$py" -u value_test.py "$model" results
fi
printf 'DONE\n'
