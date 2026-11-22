#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
test "$(tail -n 1 test.log)" = DONE
sha256sum -c results/GEOMETRY_CODE_LOCK.sha256
bash gpu_budget.sh geometry equal_dose.py "$model" "$corpus" results "$archive"
printf 'DONE\n'
