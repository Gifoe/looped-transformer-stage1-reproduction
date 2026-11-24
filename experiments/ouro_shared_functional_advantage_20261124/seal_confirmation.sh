#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
grep -qx DONE run_round3.log
grep -qx DONE run_anchored.log
lock=results/CONFIRMATION_LOCK.json
rank="$prior_transfer/results/HOLDOUT/csqa/source/s261123137/k234/controller.pt"
ret="$gen/results/FINAL_CONTROLLER.pt"
full="$gen/results/multitask_full/controller.pt"
if [[ ! -e "$lock" ]]; then
    extra=(--extra teacher_resume --extra on_policy_resume --extra anchored_policy)
    CUDA_VISIBLE_DEVICES="" "$py" freeze_eval.py --results results --data data/confirmation \
        --historical_rank "$rank" --historical_retention "$ret" --historical_mixed "$full" \
        "${extra[@]}" --out "$lock"
    sha256sum "$lock" > CONFIRMATION.sha256
fi
sha256sum -c --quiet CONFIRMATION.sha256
cp_from=$("$py" -c 'import json; x=json.load(open("results/CONFIRMATION_LOCK.json")); print(x["arm_path"][x["selected"]])')
if [[ ! -e results/FINAL_CONTROLLER.pt ]]; then cp "$cp_from" results/FINAL_CONTROLLER.pt; fi
test "$(sha256sum results/FINAL_CONTROLLER.pt | cut -d " " -f1)" = "$(sha256sum "$cp_from" | cut -d " " -f1)"
printf 'DONE %s\n' "$cp_from"
