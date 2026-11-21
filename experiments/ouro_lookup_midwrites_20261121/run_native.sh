#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-lookup-midwrites-20261121
source env.sh
sha256sum PROTOCOL.md mid_deck.py > results/NATIVE_LOCK.sha256
"$py" -u mid_deck.py "$model" results
printf 'DONE\n'
