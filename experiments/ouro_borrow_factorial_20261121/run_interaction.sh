#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-borrow-factorial-20261121
export PYTHONPATH=/root/ouro-native-state-20261121:/root/ouro-open-carry-20261121:/root/ouro-bidirectional-20261121:/root/ouro-rule-switch-20261121:/root/ouro-borrow-query-20261121:/root/ouro-borrow-transport-20261121
export OPENBLAS_NUM_THREADS=4
h=/root/autodl-tmp/ouro-cross-task-20261118
sha256sum INTERACTION.md interaction.py > results/INTERACTION_LOCK.sha256
until test "$(tail -1 natural.log)" = DONE; do
    if ! ps -p "${1:?natural PID}" -o stat= | grep -q -v Z; then echo "natural terminated"; exit 1; fi
    sleep 10
done
if test -s results/GRAFT_SUMMARY.json && "$h/venv/bin/python" -c 'import json,sys; q=json.load(open("results/GRAFT_SUMMARY.json")); sys.exit(not q["confirmation"]["forward"]["gate"])'; then
    "$h/venv/bin/python" -u interaction.py "$h/model" results
fi
printf 'DONE\n'

