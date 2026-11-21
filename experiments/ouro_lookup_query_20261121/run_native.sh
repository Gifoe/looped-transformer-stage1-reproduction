#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-lookup-query-20261121
source env.sh
sha256sum PROTOCOL.md query_deck.py > results/NATIVE_LOCK.sha256
"$py" -u query_deck.py "$model" results
printf 'DONE\n'
