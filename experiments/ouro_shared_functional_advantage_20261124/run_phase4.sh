#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for ((check=0;check<2400;check++)); do
    if grep -qx DONE run_round3.log; then break; fi
    sleep 5
done
grep -qx DONE run_round3.log
if [[ ${1:-} == --wait-anchored ]]; then
    for ((check=0;check<2400;check++)); do
        if grep -qx DONE run_anchored.log; then break; fi
        sleep 5
    done
    grep -qx DONE run_anchored.log
fi
bash seal_confirmation.sh > seal_confirmation.log 2>&1
bash run_confirmation.sh > run_confirmation.log 2>&1
bash run_postprocess.sh > run_postprocess.log 2>&1
echo DONE
