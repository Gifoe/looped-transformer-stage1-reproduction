#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-native-borrow-interaction-20261121
export PYTHONPATH=/root/ouro-native-state-20261121:/root/ouro-open-carry-20261121:/root/ouro-bidirectional-20261121:/root/ouro-rule-switch-20261121:/root/ouro-borrow-query-20261121:/root/ouro-borrow-transport-20261121:/root/ouro-borrow-factorial-20261121
export OPENBLAS_NUM_THREADS=4
h=/root/autodl-tmp/ouro-cross-task-20261118
sha256sum PATH.md mediation.py > results/PATH_LOCK.sha256
until test "$(tail -1 readout.log)" = DONE; do
    if ! ps -p "${1:?readout PID}" -o stat= | grep -q -v Z; then echo "readout terminated"; exit 1; fi
    sleep 10
done
if test -s results/READOUT_SUMMARY.json; then
    "$h/venv/bin/python" -u mediation.py "$h/model" results
fi
printf 'DONE\n'

