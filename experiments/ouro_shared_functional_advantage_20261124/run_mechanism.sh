#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
shared=results/FIT_SHARED/DIRECTIONS.pt
candidate=results/FIT_CANDIDATE/DIRECTIONS.pt
rank="$prior_transfer/results/HOLDOUT/csqa/source/s261123137/k234/controller.pt"
retention="$gen/results/FINAL_CONTROLLER.pt"
mixed="$gen/results/multitask_full/controller.pt"
arms=(--shared "$shared" --candidate "$candidate" --rank_only "$rank" --retention "$retention" --multitask "$mixed")
common=(--archive "$archive" --previous "$previous" --model "$model" --data data)
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 "$py" validate_direction.py --train results/MOMENT_FIT \
    --holdout results/MOMENT_HOLDOUT "${arms[@]}" --out results/MATRIX_HOLDOUT.json
bash control_budget.sh finite_replay finite_replay.py "${common[@]}" "${arms[@]}" --out results/FINITE_REPLAY > finite_replay.log 2>&1
bash control_budget.sh finite_early finite_replay.py "${common[@]}" "${arms[@]}" --out results/EARLY_REPLAY \
    --depths 4 --earlier-write > finite_early.log 2>&1
echo DONE
