#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-value-compose-20261121
source env.sh
sha256sum PROTOCOL.md value_deck.py > results/NATIVE_LOCK.sha256
"$py" -u value_deck.py "$model" results
printf 'DONE\n'
