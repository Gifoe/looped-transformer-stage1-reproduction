#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-native-state-20261121
h=/root/autodl-tmp/ouro-cross-task-20261118
py="$h/venv/bin/python"
model="$h/model"
export OPENBLAS_NUM_THREADS=4
wait_log=logs/causal.centered.dev.log
while ! grep -qx 'dev 48 48' "$wait_log"; do
    if grep -q '^Traceback' "$wait_log"; then exit 2; fi
    sleep 5
done
"$py" gate.py results
"$py" causal.py "$model" results BLIND.json predict
sha256sum native.py observe.py data_protocol.py behavior.py causal.py gate.py origin.py semantic.py full_tf.py PREREGISTRATION.md BLIND.json results/CANDIDATES.json results/GATE.json results/PREDICTIONS.json > results/CONFIRMATION_LOCK.sha256
if "$py" -c 'import json,sys; sys.exit(not any(v["search_writes"] for v in json.load(open("results/GATE.json"))["candidates"].values()))'; then
    "$py" -u origin.py "$model" results BLIND.json screen > logs/origin.screen.log 2>&1
    "$py" origin.py "$model" results BLIND.json freeze
fi
for p in confirm_arc confirm_external controlled; do
    "$py" -u observe.py "$model" results BLIND.json "$p" > "logs/atlas.$p.log" 2>&1
    "$py" -u causal.py "$model" results BLIND.json "$p" > "logs/causal.$p.log" 2>&1
    "$py" -u behavior.py "$model" results INPUTS.json "$p" > "logs/behavior.$p.log" 2>&1
    if [ -s results/ORIGIN_LOCK.json ]; then
        "$py" -u origin.py "$model" results BLIND.json "$p" > "logs/origin.$p.log" 2>&1
    fi
done
"$py" -u full_tf.py "$model" results BLIND.json confirm_arc > logs/full_tf.arc.log 2>&1
"$py" -u full_tf.py "$model" results BLIND.json confirm_external > logs/full_tf.external.log 2>&1
if [ -s results/ORIGIN_LOCK.json ]; then
    "$py" -u origin.py "$model" results BLIND.json confirm_external --later --limit 48 > logs/origin.later.external.log 2>&1
fi
"$py" -u semantic.py "$model" results STRUCTURE.json > logs/semantic.log 2>&1
printf 'DONE\n'

