#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-bidirectional-20261121
export PYTHONPATH=/root/ouro-native-state-20261121:/root/ouro-open-carry-20261121
export OPENBLAS_NUM_THREADS=4
h=/root/autodl-tmp/ouro-cross-task-20261118
sha256sum CHANNEL_PROTOCOL.md channels.py > results/CHANNEL_LOCK.sha256
while ! test -f results/OPERAND_SUMMARY.json; do
    if ! kill -0 "${1:?state job PID}" 2>/dev/null; then echo "state job exited"; exit 1; fi
    sleep 15
done
if "$h/venv/bin/python" -c 'import json,sys; sys.exit(not json.load(open("results/OPERAND_SUMMARY.json"))["confirmation"]["pass_primary"])'; then
    "$h/venv/bin/python" -u channels.py "$h/model" results
fi
printf 'DONE\n'

