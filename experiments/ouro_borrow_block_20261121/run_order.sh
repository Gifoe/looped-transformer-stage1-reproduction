#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-borrow-block-20261121
export PYTHONPATH=/root/ouro-native-state-20261121:/root/ouro-open-carry-20261121:/root/ouro-bidirectional-20261121:/root/ouro-rule-switch-20261121:/root/ouro-borrow-query-20261121:/root/ouro-borrow-transport-20261121:/root/ouro-borrow-factorial-20261121:/root/ouro-native-borrow-interaction-20261121
export OPENBLAS_NUM_THREADS=4
h=/root/autodl-tmp/ouro-cross-task-20261118
sha256sum ORDER.md branch_order.py > results/ORDER_LOCK.sha256
until test "$(tail -1 block.log)" = DONE; do
    if ! ps -p "${1:?block PID}" -o stat= | grep -q -v Z; then echo "block terminated"; exit 1; fi
    sleep 9
done
if test -s results/BLOCK_SUMMARY.json && "$h/venv/bin/python" -c 'import json,sys; s=json.load(open("results/BLOCK_SUMMARY.json")); sys.exit(s["repair"]["outputs"]["generated"]["K2"]<8)'; then
    "$h/venv/bin/python" -u branch_order.py "$h/model" results
fi
printf 'DONE\n'

