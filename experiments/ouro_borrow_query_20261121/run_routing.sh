#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-borrow-query-20261121
export PYTHONPATH=/root/ouro-native-state-20261121:/root/ouro-open-carry-20261121:/root/ouro-bidirectional-20261121:/root/ouro-rule-switch-20261121
export OPENBLAS_NUM_THREADS=4
h=/root/autodl-tmp/ouro-cross-task-20261118
sha256sum ROUTING_PROTOCOL.md routing.py > results/ROUTING_LOCK.sha256
until test "$(tail -1 transfer.log)" = DONE; do
    if ! ps -p "${1:?transfer PID}" -o stat= | grep -q -v Z; then echo "transfer terminated"; exit 1; fi
    sleep 10
done
if test -s results/QUERY_SUMMARY.json && "$h/venv/bin/python" -c 'import json,sys; s=json.load(open("results/QUERY_SUMMARY.json")); sys.exit(not (s["repair"]["modes"]["layer"]["joint"]>=8 and s["comparisons"]["layer_vs_layer_sham"]["joint"]["interval"][0]>0))'; then
    "$h/venv/bin/python" -u routing.py "$h/model" results
fi
printf 'DONE\n'

