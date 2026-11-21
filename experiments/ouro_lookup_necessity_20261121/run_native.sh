#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-lookup-necessity-20261121
source env.sh
sha256sum PROTOCOL.md nec_deck.py > results/NATIVE_LOCK.sha256
"$py" -u nec_deck.py "$model" results
printf 'DONE\n'
