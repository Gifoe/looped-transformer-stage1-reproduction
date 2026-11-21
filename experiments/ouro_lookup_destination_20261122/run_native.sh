#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-lookup-destination-20261122
source env.sh
sha256sum PROTOCOL.md dest_deck.py > results/NATIVE_LOCK.sha256
"$py" -u dest_deck.py "$model" results
printf 'DONE\n'
