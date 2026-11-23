#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh

if [[ ${1:-} == --wait-training ]]; then
    for ((attempt=0; attempt<360; attempt++)); do
        if awk -F '\t' '$1=="END" && $2=="csqa_train" && $6==0 {ok=1} END{exit !ok}' GPU_LEDGER.tsv; then
            break
        fi
        sleep 3
    done
    awk -F '\t' '$1=="END" && $2=="csqa_train" && $6==0 {ok=1} END{exit !ok}' GPU_LEDGER.tsv
fi

for file in results/moments/{arc,csqa}/MOMENTS.json \
    results/HOLDOUT/arc/{depth,position}/s261123137/TRAIN_PACKET.json \
    results/HOLDOUT/csqa/source/s261123137/TRAIN_PACKET.json; do
    test -s "$file"
done

common=(--archive "$archive" --previous "$previous" --model "$model")
t1=/root/ouro-unified-rank1-20261123/weights/t1.pt
forecast=results/forecast

run() {
    local name=$1 path=$2
    shift 2
    if [[ -s GPU_LEDGER.tsv ]] && awk -F '\t' -v n="$name" '$1=="END" && $2==n && $6==0 {ok=1} END{exit !ok}' GPU_LEDGER.tsv; then
        printf 'REUSE %s\n' "$name"
        return
    fi
    printf 'RUN %s\n' "$name"
    bash control_budget.sh "$name" "$path" "$@" >"$name.log" 2>&1
    printf 'COMPLETE %s\n' "$name"
}

run forecasts theory_prediction.py "${common[@]}" --historical-rank1 "$t1" --data-root data --results results --out "$forecast"
lock=(--forecast-lock "$forecast/FORECAST_LOCK.json" --predictions "$forecast/PREDICTIONS.json")
arc=(--eval arc_confirm=data/arc_confirm.jsonl)
both=("${arc[@]}" --eval csqa_confirm=data/csqa_confirm.jsonl)
depth=("${common[@]}" --minimal "$prior_root" --out results/HOLDOUT)

# The primary endpoints are independent ARC-Easy and previously unused CSQA
# questions; the historical ARC-C/author CSQA validation runs come later.
run depth_confirm depth_holdout.py "${depth[@]}" --phase confirm --source arc --suite depth "${arc[@]}" "${lock[@]}"
run arc_csqa_confirm depth_holdout.py "${depth[@]}" --phase confirm --source arc --suite depth --groups k234 --eval-tag csqa_transfer \
    --eval csqa_confirm=data/csqa_confirm.jsonl "${lock[@]}"
run source_confirm depth_holdout.py "${depth[@]}" --phase confirm --source csqa --suite source "${both[@]}" "${lock[@]}"
run position_confirm depth_holdout.py "${depth[@]}" --phase confirm --source arc --suite position "${arc[@]}" "${lock[@]}"

run state_horizon_confirm state_horizon.py "${common[@]}" --rank1 "$t1" --data data/arc_confirm.jsonl \
    --moment-root results/moments/arc --n 96 --forecast-lock "$forecast/FORECAST_LOCK.json" --evaluation-name arc_confirm \
    --out results/state_horizon/arc_confirm

run transport_confirm control_transport.py "${common[@]}" --minimal "$prior_root" --rank1 "$t1" \
    --cohorts data/MANIFEST.json --cohort arc_confirm --out results/D_TRANSPORT/arc_confirm --branch-controls

printf 'DONE\n'
