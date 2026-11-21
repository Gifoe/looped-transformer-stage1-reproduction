#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-lookup-conflict-20261121
source env.sh
sha256sum PROTOCOL.md conf_pair.py > results/PAIR_LOCK.sha256
until test "$(tail -1 native.log)" = DONE; do
    if ! ps -p "${1:?native PID}" -o stat= | grep -q -v Z; then echo "native terminated"; exit 1; fi
    sleep 8
done
"$py" -u conf_pair.py "$model" results
printf 'DONE\n'
