#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
c=(--archive "$archive" --previous "$previous" --model "$model")
weights="$transport_archive/results/HOLDOUT"
data="$transport_archive/data"
phase() {
    local tag=$1 data_file=$2 weight_file=$3
    shift 3
    bash control_budget.sh "$tag" e2_signed.py "${c[@]}" --rank1 "$weight_file" --data "$data_file" \
        --out "results/$tag" "$@" > "$tag.log" 2>&1
    echo "DONE $tag"
}
phase ARC_T1 "$data/arc_discovery.jsonl" "$t1" --n 48 --cell 1:3 --cell 1:4 --cell 2:4 --cell 2:5 --cell 4:6
phase CSQA_T1 "$data/csqa_discovery.jsonl" "$t1" --n 48 --cell 1:3 --cell 1:4 --cell 2:4 --cell 2:5 --cell 4:6
phase ARC_K2ONLY "$data/arc_discovery.jsonl" "$weights/arc/depth/s261123137/k2/controller.pt" \
    --n 32 --cell 1:3 --cell 2:4 --cell 4:6
phase ARC_QASC_RECHECK "$data/optional/arc_round2_confirm.jsonl" "$weights/qasc/source/s261123137/k234/controller.pt" \
    --n 36 --cell 1:3 --cell 2:4 --cell 4:6
echo DONE
