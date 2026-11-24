#!/usr/bin/env bash
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
external_root=${OURO_EXTERNAL_ROOT:-/root/ouro-external-data-joint-training-20261124}
cd "$external_root"
source external_env.sh
cd "$study"
export PYTHONPATH="$study:$PYTHONPATH"

while ! grep -qx DONE run_forecast.log; do
    worker=$(cat run_forecast.pid)
    if [[ ! -e "/proc/$worker/cwd" ]]; then
        echo 'Forecast worker stopped before confirmation completed' >&2
        exit 3
    fi
    sleep 10
done
exec 8>"$gpu_lock"
flock 8
forecast=results/forecast_candidate
if [[ ! -s "$forecast/AUDIT.json" ]]; then
    "$py" -u score_forecast.py --forecast "$forecast/FORECAST.json" --results "$forecast/confirmation" \
        --out "$forecast/AUDIT.json" > logs/forecast_audit.log 2>&1
fi
common=(--archive "$archive" --previous "$previous" --model "$model" --data data/pools_complete
        --moments results/filtered_moments --forecast "$forecast")
t1="$controllability/results/weights/rank1_t1.pt"
t2="$controllability/results/weights/rank1_t2.pt"
mixed="$advantage/results/fits/mixed/controller.pt"
readarray -t roster < <("$py" - "$forecast/FORECAST.json" <<'PY'
import json
import sys
from pathlib import Path

data=json.loads(Path(sys.argv[1]).read_text())
print(min(x['n_fit'] for x in data['sources'].values()))
print(*data['source_order'],sep='\n')
PY
)
limit=${roster[0]}
weights=(--weight "t1=$t1" --weight "t2=$t2" --weight "mixed=$mixed")
for source in "${roster[@]:1}"; do
    dest="results/fits/source_$source"
    if [[ ! -s "$dest/TRAIN.json" ]]; then
        printf 'START matched source %s (%s questions) %s\n' "$source" "$limit" "$(date -Is)"
        "$py" -u train_identification.py "${common[@]}" --arm uniform --include "$source" \
            --limit "$limit" --out "$dest" > "logs/fit_source_$source.log" 2>&1
    fi
    weights+=(--weight "source_$source=$dest/controller.pt")
done
for label in uniform ordinary loop warm_uniform warm_loop; do
    dest="results/fits/$label"
    initialization=()
    if [[ "$label" == warm_* ]]; then initialization=(--initialize "$t1"); fi
    if [[ ! -s "$dest/TRAIN.json" ]]; then
        echo START "$label" "$(date -Is)"
        "$py" -u train_identification.py "${common[@]}" --arm "${label#warm_}" \
            "${initialization[@]}" --out "$dest" > "logs/fit_$label.log" 2>&1
    fi
    weights+=(--weight "$label=$dest/controller.pt")
done
if [[ ! -s results/dev/SELECTION.json ]]; then
    "$py" -u evaluate_development.py --archive "$archive" --previous "$previous" --model "$model" \
        --panel "$external_root/data/panel" --stage2 "$external_root/mixed_full/archive" \
        --mmlu "$feedback/stage2/zero_shot/mmlu_cache" --unified "$unified" --out results/dev \
        "${weights[@]}" > logs/development.log 2>&1
fi
printf 'DONE\n'
