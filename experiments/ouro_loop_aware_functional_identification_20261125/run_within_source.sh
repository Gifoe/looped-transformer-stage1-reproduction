#!/usr/bin/env bash
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
joint=${OURO_EXTERNAL_ROOT:-/root/ouro-external-data-joint-training-20261124}
cd "$joint"
source external_env.sh
cd "$study"
export PYTHONPATH="$study:$PYTHONPATH"
mkdir -p logs
while ! grep -qx DONE run_fit_refinement.log; do
    worker=$(cat run_fit_refinement.pid)
    if [[ ! -e "/proc/$worker/cwd" ]]; then
        echo 'The matched compute stage stopped' >&2
        exit 3
    fi
    sleep 15
done
exec 8>"$gpu_lock"
flock 8
test -s results/dev_refinement/SELECTION.json
runtime=(--archive "$archive" --previous "$previous" --model "$model")
common=("${runtime[@]}" --data data/pools_complete --moments results/filtered_moments --forecast results/forecast_candidate)
t1="$controllability/results/weights/rank1_t1.pt"
t2="$controllability/results/weights/rank1_t2.pt"
mixed="$advantage/results/fits/mixed/controller.pt"
weights=(--weight "t1=$t1" --weight "t2=$t2" --weight "mixed=$mixed")
for domain in arc_challenge csqa; do
    for policy in uniform high low; do
        label="within_${domain}_$policy"
        dst="results/fits/$label"
        if [[ ! -s "$dst/TRAIN.json" ]]; then
            echo START "$label" "$(date -Is)"
            "$py" -u train_within_source.py "${common[@]}" --source "$domain" --policy "$policy" \
                --out "$dst" > "logs/fit_$label.log" 2>&1
        fi
        weights+=(--weight "$label=$dst/controller.pt")
    done
done
if [[ ! -s results/dev_within_source/SELECTION.json ]]; then
    "$py" -u evaluate_development.py "${runtime[@]}" --panel "$joint/data/panel" \
        --stage2 "$joint/mixed_full/archive" --mmlu "$feedback/stage2/zero_shot/mmlu_cache" \
        --unified "$unified" --reuse results/dev --out results/dev_within_source "${weights[@]}" > logs/development_within_source.log 2>&1
fi
"$py" -u score_source_training.py --forecast results/forecast_candidate/FORECAST.json \
    --development results/dev/SELECTION.json --work-development results/dev_refinement/SELECTION.json \
    --out results/forecast_candidate/TRAINED_VALUE.json
printf 'DONE\n'
