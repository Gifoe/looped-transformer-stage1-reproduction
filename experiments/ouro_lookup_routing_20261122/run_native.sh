#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-lookup-routing-20261122
source env.sh
sha256sum PROTOCOL.md route_deck.py > results/NATIVE_LOCK.sha256
"$py" -u route_deck.py "$model" results
printf 'DONE\n'
