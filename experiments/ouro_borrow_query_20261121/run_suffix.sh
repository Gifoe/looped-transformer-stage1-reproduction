#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-borrow-query-20261121
export PYTHONPATH=/root/ouro-native-state-20261121:/root/ouro-open-carry-20261121:/root/ouro-bidirectional-20261121:/root/ouro-rule-switch-20261121
export OPENBLAS_NUM_THREADS=4
h=/root/autodl-tmp/ouro-cross-task-20261118
sha256sum SUFFIX_PROTOCOL.md suffix.py > results/SUFFIX_LOCK.sha256
until test "$(tail -1 routing.log)" = DONE; do
    if ! ps -p "${1:?routing PID}" -o stat= | grep -q -v Z; then echo "routing terminated"; exit 1; fi
    sleep 10
done
if test -s results/CONTEXT_SUMMARY.json && "$h/venv/bin/python" -c 'import json,sys; s=json.load(open("results/CONTEXT_SUMMARY.json"))["repair"]; sys.exit(not s["n"]-s["arms"]["full"]["K1_correct"]>=8)'; then
    "$h/venv/bin/python" -u suffix.py "$h/model" results
fi
printf 'DONE\n'

