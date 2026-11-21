#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-borrow-edge-20261121
source env.sh
sha256sum EXTENSION.md extension.py run_extension.sh > results/EXTENSION_LOCK.sha256
"$py" extension.py
until test "$(tail -1 edge.log)" = DONE; do
    if ! ps -p "${1:?primary edge PID}" -o stat= | grep -q -v Z; then echo "primary edge terminated"; exit 1; fi
    sleep 6
done
"$py" -u edge_deck.py "$model" results/extension
"$py" -u edge.py "$model" results/extension
printf 'DONE\n'
