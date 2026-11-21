#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-rule-switch-20261121
export PYTHONPATH=/root/ouro-native-state-20261121:/root/ouro-open-carry-20261121:/root/ouro-bidirectional-20261121
export OPENBLAS_NUM_THREADS=4
h=/root/autodl-tmp/ouro-cross-task-20261118
sha256sum BOUNDARY_PROTOCOL.md boundary_control.py > results/BOUNDARY_LOCK.sha256
until test "$(tail -1 natural.log)" = DONE; do
    if ! ps -p "${1:?natural PID}" -o stat= | grep -q -v Z; then echo "natural terminated"; exit 1; fi
    sleep 10
done
"$h/venv/bin/python" -u boundary_control.py "$h/model" /root/ouro-open-carry-20261121/MANIFEST.json /root/ouro-carry-boundary-20261121/results/MANIFEST.json /root/ouro-bidirectional-20261121/results/DECK.json results
printf 'DONE\n'

