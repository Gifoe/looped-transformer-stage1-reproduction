#!/usr/bin/env bash
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
joint=${OURO_EXTERNAL_ROOT:-/root/ouro-external-data-joint-training-20261124}
cd "$joint"
source external_env.sh
cd "$study"
export PYTHONPATH="$study:$PYTHONPATH"
mkdir -p logs

test -s results/filtered_moments/social_iqa/candidate/k4/fold_3.jsonl
test -s results/filtered_moments/gsm8k/generation/k4/fold_3.jsonl
forecast=results/forecast_eight_sources
if [[ ! -s "$forecast/FORECAST.json" ]]; then
    "$py" -u freeze_forecast.py --moments results/filtered_moments --data data/pools_complete \
        --sources arc_challenge:candidate csqa:candidate obqa:candidate qasc:candidate scienceqa:candidate \
                  boolq:candidate social_iqa:candidate gsm8k:generation \
        --scale-weight "$controllability/results/weights/rank1_t1.pt" --out "$forecast"
fi
test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | xargs)" -le 8500
"$py" -u test_forecast.py --archive "$archive" --previous "$previous" --model "$model" \
    --data data/pools_complete --forecast "$forecast" --depths 4 3 5 --n 48 \
    --reuse results/forecast_candidate --out "$forecast/confirmation"
"$py" -u score_forecast.py --forecast "$forecast/FORECAST.json" --results "$forecast/confirmation" \
    --out "$forecast/AUDIT.json"
printf 'DONE\n'
