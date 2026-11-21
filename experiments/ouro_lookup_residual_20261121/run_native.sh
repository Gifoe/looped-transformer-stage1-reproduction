#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-lookup-residual-20261121
source env.sh
sha256sum PROTOCOL.md residual_deck.py > results/NATIVE_LOCK.sha256
"$py" -u residual_deck.py "$model" results
printf 'DONE\n'
