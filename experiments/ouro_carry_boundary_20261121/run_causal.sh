#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-carry-boundary-20261121
export PYTHONPATH=/root/ouro-native-state-20261121:/root/ouro-open-carry-20261121
export OPENBLAS_NUM_THREADS=4
h=/root/autodl-tmp/ouro-cross-task-20261118
sha256sum CAUSAL_PROTOCOL.md causal.py analysis.py > results/CAUSAL_LOCK.sha256
while ! test -f results/NATIVE_SUMMARY.json; do
    if ! kill -0 "${1:?native job PID}" 2>/dev/null; then echo "behavior job exited early"; exit 1; fi
    sleep 15
done
"$h/venv/bin/python" -u analysis.py behavior results
if "$h/venv/bin/python" -c 'import json,sys; sys.exit(not json.load(open("results/BEHAVIOR_GATE.json"))["pass"])'; then
    "$h/venv/bin/python" -u causal.py screen "$h/model" results
    "$h/venv/bin/python" -u causal.py confirm "$h/model" results
    if compgen -G 'results/CONFIRMATION_RAW_*.json' >/dev/null; then
        "$h/venv/bin/python" -u analysis.py causal results
    fi
fi
printf 'DONE\n'

