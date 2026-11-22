#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
sha256sum minimal.py train.py > results/TRAIN_CODE_LOCK.sha256
bash gpu_budget.sh screen train.py screen "$model" "$archive" "$previous" "$corpus" results
printf 'DONE\n'
