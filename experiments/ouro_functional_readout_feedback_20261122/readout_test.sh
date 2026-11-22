#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
test "$(tail -n 1 dual.log)" = DONE
test -s results/CALIBRATION_FREEZE.json
sha256sum -c results/DUAL_CODE_LOCK.sha256
sha256sum readout_eval.py calibration.py results/READOUT_FREEZE.json results/CALIBRATION_FREEZE.json > results/READOUT_TEST_LOCK.sha256
bash gpu_budget.sh arc_readout readout_eval.py "$model" "$previous" "$archive" "$corpus" results
bash gpu_budget.sh openbook_readout readout_eval.py "$model" "$previous" "$archive" "$corpus" results --external external/openbookqa.parquet
printf 'DONE\n'
