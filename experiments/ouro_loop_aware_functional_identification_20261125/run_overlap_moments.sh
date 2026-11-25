#!/usr/bin/env bash
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
joint=${OURO_EXTERNAL_ROOT:-/root/ouro-external-data-joint-training-20261124}
cd "$joint"
source external_env.sh
cd "$study"
export PYTHONPATH="$study:$PYTHONPATH"
mkdir -p logs

runtime=(--archive "$archive" --previous "$previous" --model "$model" --full-context)
operators=(--operator "t1=$controllability/results/weights/rank1_t1.pt"
           --operator "t2=$controllability/results/weights/rank1_t2.pt"
           --operator "mixed=$advantage/results/fits/mixed/controller.pt")

# This may share the card with the ongoing development decoder. Formal latency
# measurements still take the exclusive GPU lock.
for source in social_iqa gsm8k; do
    if grep -qx DONE run_fits.log; then
        echo 'Development finished. The remaining sources retain their normal GPU queue.'
        exit 0
    fi
    mode=candidate
    if [[ "$source" != social_iqa ]]; then mode=generation; fi
    if [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | xargs) -gt 8500 ]]; then
        echo 'Insufficient VRAM for concurrent collection' >&2
        exit 3
    fi
    echo START "$source" "$(date -Is)"
    "$py" -u -c 'import torch; import collect_identification as c; torch.cuda.set_per_process_memory_fraction(.58); c.collect(c.parser().parse_args())' \
        "${runtime[@]}" "${operators[@]}" \
        --input "data/pools_complete/$source.jsonl" --dataset "$source" --mode "$mode" \
        --depth 4 --batch 4 --token-budget 2304 --out "results/filtered_moments/$source/$mode/k4" \
        > "logs/moments_overlap_${source}_k4.log" 2>&1
done
printf 'DONE\n'
