#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-rule-switch-20261121
export PYTHONPATH=/root/ouro-native-state-20261121:/root/ouro-open-carry-20261121:/root/ouro-bidirectional-20261121
export OPENBLAS_NUM_THREADS=4
h=/root/autodl-tmp/ouro-cross-task-20261118
sha256sum PATH_PROTOCOL.md pathway.py > results/PATH_LOCK.sha256
while ! test -s results/GRAFT_SUMMARY.json; do
    if test "$(tail -1 graft.log)" = DONE && ! test -s results/GRAFT_SUMMARY.json; then printf 'DONE\n'; exit 0; fi
    if ! ps -p "${1:?graft PID}" -o stat= | grep -q -v Z; then echo "graft terminated"; exit 1; fi
    sleep 10
done
if "$h/venv/bin/python" -c 'import json,sys; sys.exit(not json.load(open("results/GRAFT_SUMMARY.json"))["confirmation"]["gate"])'; then
    "$h/venv/bin/python" -u pathway.py "$h/model" results
fi
printf 'DONE\n'

