#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-boundary-factor-20261121
source env.sh
sha256sum PROTOCOL.md boundary_deck.py > results/NATIVE_LOCK.sha256
"$py" -u boundary_deck.py "$model" results
printf 'DONE\n'
