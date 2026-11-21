#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-borrow-origin-20261121
export PYTHONPATH=/root/ouro-native-state-20261121:/root/ouro-open-carry-20261121:/root/ouro-bidirectional-20261121:/root/ouro-rule-switch-20261121:/root/ouro-borrow-query-20261121:/root/ouro-borrow-transport-20261121:/root/ouro-borrow-factorial-20261121:/root/ouro-native-borrow-interaction-20261121:/root/ouro-borrow-block-20261121
export OPENBLAS_NUM_THREADS=4
h=/root/autodl-tmp/ouro-cross-task-20261118
sha256sum PROTOCOL.md origin_deck.py > results/NATIVE_LOCK.sha256
"$h/venv/bin/python" -u origin_deck.py "$h/model" results
printf 'DONE\n'
