#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for ((check=0;check<1200;check++)); do
    if grep -qx DONE run_mechanism.log; then break; fi
    sleep 5
done
grep -qx DONE run_mechanism.log
bash run_fitting.sh > run_fitting.log 2>&1
bash run_development.sh > run_development.log 2>&1
echo DONE
