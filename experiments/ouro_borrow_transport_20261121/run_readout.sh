#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-borrow-transport-20261121
export PYTHONPATH=/root/ouro-native-state-20261121:/root/ouro-open-carry-20261121:/root/ouro-bidirectional-20261121:/root/ouro-rule-switch-20261121:/root/ouro-borrow-query-20261121
export OPENBLAS_NUM_THREADS=4
h=/root/autodl-tmp/ouro-cross-task-20261118
sha256sum READOUT_PROTOCOL.md readout_check.py > results/READOUT_LOCK.sha256
until test "$(tail -1 mechanism.log)" = DONE; do
    if ! ps -p "${1:?mechanism PID}" -o stat= | grep -q -v Z; then echo "mechanism terminated"; exit 1; fi
    sleep 10
done
if test -s results/MATCHING.json && "$h/venv/bin/python" -c 'import json,sys; sys.exit(not json.load(open("results/MATCHING.json"))["gate"])'; then
    "$h/venv/bin/python" -u readout_check.py "$h/model" results
fi
printf 'DONE\n'

