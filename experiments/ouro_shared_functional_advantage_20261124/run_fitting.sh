#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
rank="$prior_transfer/results/HOLDOUT/csqa/source/s261123137/k234/controller.pt"
mixed="$gen/results/multitask_full/controller.pt"
common=(--archive "$archive" --previous "$previous" --model "$model" --data data --moments results/MOMENT_FIT)
grep -qx DONE run_mechanism.log
CUDA_VISIBLE_DEVICES="" "$py" analyze_mechanism.py --fit results/MOMENT_FIT --late results/FINITE_REPLAY \
    --early results/EARLY_REPLAY --out results/FINITE_SUMMARY.json

if [[ ! -s results/PREFIX_CAUSALITY/RECEIPT.json ]]; then
    bash control_budget.sh prefix_causality prefix_causality.py --archive "$archive" --previous "$previous" \
        --model "$model" --data data/gsm8k_moment_holdout.jsonl --retention "$gen/results/FINAL_CONTROLLER.pt" \
        --out results/PREFIX_CAUSALITY > prefix_causality.log 2>&1
fi
if [[ ! -s results/SAMPLE_EFFICIENCY.json ]]; then
    for n in 8 16 32; do
        bash control_budget.sh "sample_moments_$n" collect_moments.py --archive "$archive" --previous "$previous" \
            --model "$model" --data data --fold moment_fit --candidate-only --limit "$n" \
            --out "results/SAMPLE_MOMENTS_$n" > "sample_moments_$n.log" 2>&1
    done
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 "$py" sample_size.py --results results --fit results/MOMENT_FIT \
        --holdout results/MOMENT_HOLDOUT --out results/SAMPLE_EFFICIENCY.json
fi

fit() {
    local arm=$1 objective=$2
    shift 2
    if [[ ! -s "results/fits/$arm/TRAIN.json" ]]; then
        bash control_budget.sh "train_$arm" train_advantage.py "${common[@]}" --out "results/fits/$arm" \
            --objective "$objective" "$@" > "train_${arm}.log" 2>&1
    fi
}

fit ranking ranking
fit generation generation
fit mixed mixed
fit mean_advantage mean_advantage
fit worst_advantage worst_advantage
fit worst_rank_init worst_advantage --initial "$rank"
fit worst_mixed_init worst_advantage --initial "$mixed"
echo DONE
