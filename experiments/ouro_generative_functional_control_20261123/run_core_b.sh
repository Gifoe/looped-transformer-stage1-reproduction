#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
mkdir -p data results
common=(--archive "$archive" --previous "$previous" --model "$model" --data data/csqa_matched.jsonl)
reference="$prior_transfer/results/HOLDOUT/csqa/source/s261123137/k234/controller.pt"
if [[ ! -s data/csqa_matched.jsonl ]]; then
    CUDA_VISIBLE_DEVICES="" "$py" prepare_baseline.py "$prior_transfer" data/csqa_matched.jsonl
fi
if [[ ! -s results/PREFLIGHT/PARITY.json ]]; then
    bash control_budget.sh parity train_generative.py "${common[@]}" --phase parity --initial "$reference" \
        --out results/PREFLIGHT > parity.log 2>&1
fi
for objective in gen_only gen_rank; do
    aux=0
    if [[ "$objective" == gen_rank ]]; then aux=.25; fi
    if [[ ! -s "results/$objective/TRAIN.json" ]]; then
        bash control_budget.sh "$objective" train_generative.py "${common[@]}" --rank-weight "$aux" \
            --out "results/$objective" --preserve-order > "$objective.log" 2>&1
    fi
    echo "DONE $objective"
done
echo DONE
