#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-rule-switch-20261121
export PYTHONPATH=/root/ouro-native-state-20261121:/root/ouro-open-carry-20261121:/root/ouro-bidirectional-20261121
export OPENBLAS_NUM_THREADS=4
h=/root/autodl-tmp/ouro-cross-task-20261118
mkdir -p results
sha256sum PROTOCOL.md rule.py > results/NATIVE_LOCK.sha256
"$h/venv/bin/python" -u rule.py "$h/model" /root/ouro-open-carry-20261121/MANIFEST.json /root/ouro-carry-boundary-20261121/results/MANIFEST.json /root/ouro-bidirectional-20261121/results/DECK.json results
printf 'DONE\n'

