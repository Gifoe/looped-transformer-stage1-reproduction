#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
test "$(tail -n 1 features.log)" = DONE
sha256sum amplify.py evaluate.py > results/TRAIN_CODE_LOCK.sha256
bash gpu_budget.sh training amplify.py "$model" features results "$archive"
printf 'DONE\n'
