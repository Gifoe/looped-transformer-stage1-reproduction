#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source external_env.sh
test -s data/MANIFEST.json
test -s data/SCHEDULE_PREFLIGHT.json
if [[ ${1:-} == --wait-shared ]]; then
    while ! grep -qx DONE "$advantage/run_confirmation.log"; do
        pid=$(cat "$advantage/resumed_phase4.pid")
        state=$(ps -p "$pid" -o stat= | tr -d ' ')
        [[ -n "$state" && "$state" != Z* ]]
        sleep 15
    done
fi
common=(--archive "$archive" --previous "$previous" --model "$model" --data data)
seed=261124137
mkdir -p logs
bash gpu_job.sh smoke_B train_external.py "${common[@]}" --recipe B --seed "$seed" --through-epoch 1 \
    --smoke-updates 3 --out results/smoke/B > logs/smoke_B.log 2>&1
for recipe in A B; do
    bash gpu_job.sh "fit_${recipe}_${seed}" train_external.py "${common[@]}" --recipe "$recipe" --seed "$seed" \
        --through-epoch 3 --out "results/fits/$recipe/$seed" > "logs/fit_${recipe}_${seed}.log" 2>&1
done
echo DONE
