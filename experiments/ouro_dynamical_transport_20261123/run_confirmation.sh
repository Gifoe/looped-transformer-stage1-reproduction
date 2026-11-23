#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
test "$(tail -1 run_e3_discovery.log)" = DONE
c=(--archive "$archive" --previous "$previous" --model "$model")
weights="$transport_archive/results/HOLDOUT"
qasc="$weights/qasc/source/s261123137/k234/controller.pt"
k2="$weights/arc/depth/s261123137/k2/controller.pt"
seal=results/FORECAST_LOCK.json
if [[ ! -s "$seal" ]]; then
    CUDA_VISIBLE_DEVICES="" "$py" seal_forecast.py --results results --data data --t1 "$t1" --k2 "$k2" --qasc "$qasc" --out "$seal"
    sha256sum "$seal" > FORECAST_LOCK.sha256
fi
sha256sum -c --quiet FORECAST_LOCK.sha256
run() {
    local tag=$1 entry=$2
    shift 2
    if awk -F '\t' -v stage="$tag" '$1=="END"&&$2==stage&&$6==0{found=1} END{exit !found}' GPU_LEDGER.tsv; then
        echo "REUSE $tag"
        return
    fi
    bash control_budget.sh "$tag" "$entry" "${c[@]}" "$@" --out "results/$tag" > "$tag.log" 2>&1
    echo "DONE $tag"
}

for domain in arc csqa; do
    file="$PWD/data/${domain}_confirm.jsonl"
    run "E1_FRESH_$domain" e1_confirm.py --t1 "$t1" --transport "$transport_archive" \
        --forecast-lock "$seal" --domain "$domain" --data "$file"
    run "E2_FRESH_$domain" e2_signed.py --rank1 "$t1" --data "$file" --n 64 --cell 1:3 --cell 2:4 --cell 4:6
    run "HORIZON_$domain" horizon_adjoint.py --rank1 "$t1" --data "$file" --domain "$domain" \
        --forecast-lock "$seal" --n 72
    run "E3_FRESH_$domain" e3_paths.py --transport "$transport_archive" --rank1 "$t1" --data "$file" \
        --domain "$domain" --cell 1:4 --cell 2:4 --n 72 --batch 6 --forecast-lock "$seal"
done
run E2_FRESH_QASC e2_signed.py --rank1 "$qasc" --data "$PWD/data/arc_confirm.jsonl" --n 72 \
    --cell 1:3 --cell 2:4 --cell 4:6
run E2_FRESH_K2ONLY e2_signed.py --rank1 "$k2" --data "$PWD/data/arc_confirm.jsonl" --n 72 --cell 4:6
echo DONE
