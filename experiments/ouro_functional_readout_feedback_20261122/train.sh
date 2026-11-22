#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
sha256sum dual.py bridge.py > results/DUAL_CODE_LOCK.sha256
bash gpu_budget.sh dual dual.py "$model" "$previous" "$archive" results
printf 'DONE\n'
