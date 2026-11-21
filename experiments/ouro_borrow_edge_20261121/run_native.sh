#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-borrow-edge-20261121
source env.sh
sha256sum PROTOCOL.md edge_deck.py > results/NATIVE_LOCK.sha256
"$py" -u edge_deck.py "$model" results
printf 'DONE\n'
