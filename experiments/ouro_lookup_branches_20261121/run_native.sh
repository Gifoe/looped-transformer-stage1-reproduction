#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-lookup-branches-20261121
source env.sh
sha256sum PROTOCOL.md branches_deck.py > results/NATIVE_LOCK.sha256
"$py" -u branches_deck.py "$model" results
printf 'DONE\n'
