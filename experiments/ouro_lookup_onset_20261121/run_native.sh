#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-lookup-onset-20261121
source env.sh
sha256sum PROTOCOL.md onset_deck.py > results/NATIVE_LOCK.sha256
"$py" -u onset_deck.py "$model" results
printf 'DONE\n'
