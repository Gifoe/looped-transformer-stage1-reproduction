#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-borrow-transport-20261121
export PYTHONPATH=/root/ouro-native-state-20261121:/root/ouro-open-carry-20261121:/root/ouro-bidirectional-20261121:/root/ouro-rule-switch-20261121:/root/ouro-borrow-query-20261121
export OPENBLAS_NUM_THREADS=4
h=/root/autodl-tmp/ouro-cross-task-20261118
sha256sum TRANSPORT.md mechanism.py > results/MECHANISM_LOCK.sha256
until test -s results/NATIVE_SUMMARY.json; do
    if ! ps -p "${1:?native PID}" -o stat= | grep -q -v Z; then echo "native terminated"; exit 1; fi
    sleep 10
done
if "$h/venv/bin/python" -c 'import json,sys; sys.exit(not json.load(open("results/NATIVE_SUMMARY.json"))["gate"])'; then
    "$h/venv/bin/python" -u mechanism.py "$h/model" results
fi
printf 'DONE\n'

