#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh

for ((i=0; i<600; i++)); do
    if grep -qx DONE evaluate_core.log && [[ -s data/optional/ROUND2_MANIFEST.json ]]; then break; fi
    sleep 3
done
grep -qx DONE evaluate_core.log
test -s data/optional/ROUND2_MANIFEST.json

common=(--archive "$archive" --previous "$previous" --model "$model")
ctl=("${common[@]}" --minimal "$prior_root" --out results/HOLDOUT)

run() {
    local name="$1" path="$2"
    shift 2
    if awk -F '\t' -v n="$name" '$1=="END" && $2==n && $6==0 {ok=1} END{exit !ok}' GPU_LEDGER.tsv; then
        printf 'REUSE %s\n' "$name"
        return
    fi
    printf 'RUN %s\n' "$name"
    bash control_budget.sh "$name" "$path" "$@" >"$name.log" 2>&1
    printf 'COMPLETE %s\n' "$name"
}

data_for() {
    case "$1" in
        arc|csqa) printf 'data/%s_train.jsonl\n' "$1" ;;
        obqa|qasc) printf 'data/optional/%s_train.jsonl\n' "$1" ;;
        mixed) printf 'data/arc_csqa_mixed_train.jsonl\n' ;;
    esac
}

trainer_for() {
    if [[ "$1" == qasc ]]; then
        printf 'depth_holdout_qasc.py\n'
    else
        printf 'depth_holdout.py\n'
    fi
}

for source in obqa qasc mixed; do
    file=$(data_for "$source")
    run "${source}_moments" functional_operator.py "${common[@]}" --train-data "$file" --n 96 --out "results/moments/$source"
done

for seed in 261123137 261123271; do
    for source in arc csqa obqa qasc mixed; do
        if [[ "$seed" == 261123137 && ( "$source" == arc || "$source" == csqa ) ]]; then continue; fi
        file=$(data_for "$source")
        run "r2_fit_${source}_${seed}" "$(trainer_for "$source")" "${ctl[@]}" --phase train --source "$source" --suite source \
            --train-data "$file" --seed "$seed"
    done
done

forecast=results/forecast_round2
run r2_forecasts theory_prediction.py "${common[@]}" --historical-rank1 /root/ouro-unified-rank1-20261123/weights/t1.pt \
    --data-root data --results results --out "$forecast" --round2
confirm=(--eval arc_round2_confirm=data/optional/arc_round2_confirm.jsonl \
    --eval csqa_round2_confirm=data/optional/csqa_round2_confirm.jsonl \
    --forecast-lock "$forecast/FORECAST_LOCK.json" --predictions "$forecast/PREDICTIONS.json")

for seed in 261123137 261123271; do
    for source in arc csqa obqa qasc mixed; do
        suite=source
        if [[ "$seed" == 261123137 && "$source" == arc ]]; then suite=depth; fi
        run "r2_eval_${source}_${seed}" "$(trainer_for "$source")" "${ctl[@]}" --phase confirm --source "$source" --suite "$suite" \
            --groups k234 --seed "$seed" --eval-tag r2 "${confirm[@]}"
    done
done
printf 'DONE\n'
