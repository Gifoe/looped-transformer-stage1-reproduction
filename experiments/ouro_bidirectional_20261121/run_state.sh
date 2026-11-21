#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-bidirectional-20261121
export PYTHONPATH=/root/ouro-native-state-20261121:/root/ouro-open-carry-20261121
export OPENBLAS_NUM_THREADS=4
h=/root/autodl-tmp/ouro-cross-task-20261118
sha256sum STATE_PROTOCOL.md state.py OPERAND_PROTOCOL.md operand.py > results/STATE_LOCK.sha256
while ! test -f results/NATIVE_SUMMARY.json; do
    if ! kill -0 "${1:?native job PID}" 2>/dev/null; then echo "native job exited"; exit 1; fi
    sleep 15
done
if "$h/venv/bin/python" -c 'import json,sys; sys.exit(not json.load(open("results/NATIVE_SUMMARY.json"))["confirmation"]["pass_gate"])'; then
    "$h/venv/bin/python" -u state.py "$h/model" results
fi
"$h/venv/bin/python" -u operand.py "$h/model" results
printf 'DONE\n'

