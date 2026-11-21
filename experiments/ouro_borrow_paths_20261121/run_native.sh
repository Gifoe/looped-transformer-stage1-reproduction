#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-borrow-paths-20261121
source env.sh
sha256sum PROTOCOL.md path_deck.py > results/NATIVE_LOCK.sha256
"$py" -u path_deck.py "$model" results
printf 'DONE\n'
