#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-native-state-20261121
h=/root/autodl-tmp/ouro-cross-task-20261118
export OPENBLAS_NUM_THREADS=4
while ! grep -qx DONE confirm.log; do
    if grep -q '^Traceback' confirm.log; then exit 2; fi
    sleep 10
done
"$h/venv/bin/python" -u origin_formation.py "$h/model" results BLIND.json FORMATION_PROTOCOL.md > logs/origin.formation.log 2>&1
printf 'DONE\n'

