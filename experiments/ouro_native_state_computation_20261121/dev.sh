#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-native-state-20261121
h=/root/autodl-tmp/ouro-cross-task-20261118
export OPENBLAS_NUM_THREADS=4
while ! grep -qx DONE early.log; do
    ls logs/atlas.dev.log >/dev/null 2>&1 || { sleep 5; continue; }
    if grep -q '^Traceback' logs/atlas.dev.log; then exit 2; fi
    sleep 5
done
"$h/venv/bin/python" -u causal.py "$h/model" results BLIND.json dev --limit 48 > logs/causal.dev.log 2>&1
"$h/venv/bin/python" -u behavior.py "$h/model" results INPUTS.json dev --max-new 12 > logs/behavior.dev.log 2>&1
printf 'DONE\n'

