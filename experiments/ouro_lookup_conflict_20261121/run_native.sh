#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-lookup-conflict-20261121
source env.sh
sha256sum PROTOCOL.md conf_deck.py > results/NATIVE_LOCK.sha256
"$py" -u conf_deck.py "$model" results
printf 'DONE\n'
