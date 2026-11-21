#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-lookup-midwrites-20261121
source env.sh
sha256sum PROTOCOL.md mid_test.py mid_stats.py smoke.py > results/MID_LOCK.sha256
until test "$(tail -1 native.log)" = DONE; do
    if ! ps -p "${1:?native PID}" -o stat= | grep -q -v Z; then echo "native terminated"; exit 1; fi
    sleep 8
done
if "$py" -c 'import json,sys; sys.exit(not json.load(open("results/NATIVE_SUMMARY.json"))["gate"])'; then
    "$py" -u smoke.py "$model"
    "$py" -u mid_test.py "$model" results
fi
printf 'DONE\n'
