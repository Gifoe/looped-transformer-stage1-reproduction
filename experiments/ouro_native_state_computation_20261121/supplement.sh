#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-native-state-20261121
h=/root/autodl-tmp/ouro-cross-task-20261118
while ! grep -qx DONE formation.log; do
    if [ -f logs/origin.formation.log ] && grep -q '^Traceback' logs/origin.formation.log; then exit 2; fi
    sleep 10
done
"$h/venv/bin/python" -u response_controls.py "$h/model" results BLIND.json > logs/response.controls.log 2>&1
printf 'DONE\n'

