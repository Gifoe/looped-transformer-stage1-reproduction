#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-lookup-av-20261121
source env.sh
sha256sum PROTOCOL.md av_deck.py > results/NATIVE_LOCK.sha256
"$py" -u av_deck.py "$model" results
printf 'DONE\n'
