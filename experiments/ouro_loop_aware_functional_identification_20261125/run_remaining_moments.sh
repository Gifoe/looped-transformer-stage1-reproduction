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
base=(--archive "$archive" --previous "$previous" --model "$model" --full-context)
operators=(--operator "t1=$controllability/results/weights/rank1_t1.pt"
           --operator "t2=$controllability/results/weights/rank1_t2.pt"
           --operator "mixed=$advantage/results/fits/mixed/controller.pt")
for source in social_iqa race squad gsm8k; do
    mode=candidate
    if [[ "$source" == squad || "$source" == gsm8k ]]; then mode=generation; fi
    "$py" -u collect_identification.py "${base[@]}" "${operators[@]}" \
        --input "data/pools_complete/$source.jsonl" --dataset "$source" --mode "$mode" \
        --depth 4 --batch 4 --token-budget 2304 --out "results/filtered_moments/$source/$mode/k4" \
        > "logs/moments_${source}_k4.log" 2>&1
done

forecast=results/forecast_all_sources
if [[ ! -s "$forecast/FORECAST.json" ]]; then
    tags=(arc_challenge:candidate csqa:candidate obqa:candidate qasc:candidate
          scienceqa:candidate boolq:candidate social_iqa:candidate race:candidate squad:generation gsm8k:generation)
    "$py" -u freeze_forecast.py --moments results/filtered_moments --data data/pools_complete \
        --sources "${tags[@]}" --scale-weight "$controllability/results/weights/rank1_t1.pt" --out "$forecast"
fi
"$py" -u test_forecast.py --archive "$archive" --previous "$previous" --model "$model" \
    --data data/pools_complete --forecast "$forecast" --depths 4 3 5 --n 48 --out "$forecast/confirmation"
"$py" -u score_forecast.py --forecast "$forecast/FORECAST.json" --results "$forecast/confirmation" \
    --out "$forecast/AUDIT.json"
printf 'DONE\n'
