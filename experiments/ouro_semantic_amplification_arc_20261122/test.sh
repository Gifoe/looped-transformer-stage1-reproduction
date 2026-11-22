#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
test "$(tail -n 1 training.log)" = DONE
sha256sum -c results/TRAIN_CODE_LOCK.sha256
bash gpu_budget.sh test evaluate.py "$model" "$corpus" results "$archive"
printf 'DONE\n'
