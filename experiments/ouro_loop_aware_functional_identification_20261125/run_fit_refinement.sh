#!/usr/bin/env bash
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
joint=${OURO_EXTERNAL_ROOT:-/root/ouro-external-data-joint-training-20261124}
cd "$joint"
source external_env.sh
cd "$study"
export PYTHONPATH="$study:$PYTHONPATH"
mkdir -p logs
exec 8>"$gpu_lock"
flock 8

for state in results/dev/SELECTION.json results/forecast_candidate/AUDIT.json; do
    test -s "$state"
done
t1="$controllability/results/weights/rank1_t1.pt"
t2="$controllability/results/weights/rank1_t2.pt"
mixed="$advantage/results/fits/mixed/controller.pt"
forecast=results/forecast_candidate
runtime=(--archive "$archive" --previous "$previous" --model "$model")
common=("${runtime[@]}" --data data/pools_complete --moments results/filtered_moments --forecast "$forecast")
sources=(arc_challenge csqa obqa qasc scienceqa boolq)
budget=$("$py" - <<'PY'
import json
from pathlib import Path

receipts=sorted(Path('results/fits').glob('source_*/TRAIN.json'))
if len(receipts)!=6:
    raise ValueError('Cannot seal a compute budget without all six source receipts')
print(min(json.loads(file.read_text())['padded_token_loops'] for file in receipts))
PY
)
weights=(--weight "t1=$t1" --weight "t2=$t2" --weight "mixed=$mixed")
for source in "${sources[@]}"; do
    dst="results/fits/source_${source}_work"
    if [[ ! -s "$dst/TRAIN.json" ]]; then
        echo START "$source" matched_padded_token_loops "$budget" "$(date -Is)"
        "$py" -u train_identification.py "${common[@]}" --arm uniform --include "$source" \
            --limit 760 --max-work "$budget" --out "$dst" > "logs/fit_${source}_work.log" 2>&1
    fi
    weights+=(--weight "source_${source}_work=$dst/controller.pt")
done
dst=results/fits/warm_tangent
if [[ ! -s "$dst/TRAIN.json" ]]; then
    echo START warm_tangent "$(date -Is)"
    "$py" -u train_warm_tangent.py "${common[@]}" --initialize "$t1" --out "$dst" > logs/fit_warm_tangent.log 2>&1
fi
weights+=(--weight "warm_tangent=$dst/controller.pt")
for label in uniform ordinary loop warm_uniform warm_loop; do
    weights+=(--weight "$label=results/fits/$label/controller.pt")
done
if [[ ! -s results/dev_refinement/SELECTION.json ]]; then
    "$py" -u evaluate_development.py "${runtime[@]}" --panel "$joint/data/panel" \
        --stage2 "$joint/mixed_full/archive" --mmlu "$feedback/stage2/zero_shot/mmlu_cache" \
        --unified "$unified" --out results/dev_refinement "${weights[@]}" > logs/development_refinement.log 2>&1
fi
printf 'DONE\n'
