#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source external_env.sh
export CUDA_VISIBLE_DEVICES=''
shopt -s nullglob
stage="$feedback/stage2"
reserved=(
    "$prior_transfer"/inputs/arc_easy_{test,validation}.parquet
    "$prior_transfer"/inputs/upstream_arc_challenge_{test,validation}.parquet
    "$prior_transfer"/inputs/obqa_{test,validation}.parquet
    "$prior_transfer"/data/csqa_{validation,confirm,discovery}.jsonl
    "$prior_transfer"/data/optional/{arc,csqa}_round2_confirm.jsonl
    "$dynamics"/data/{arc,csqa}_confirm.jsonl
    "$dynamics"/data/path/{arc,csqa}_confirm.jsonl
    "$gen"/data/dev_*.jsonl
    "$gen"/data/test_*.jsonl
    "$gen"/inputs/{squad_validation,gsm8k_test}.parquet
    "$advantage"/data/*_development.jsonl
    "$advantage"/data/*_moment_holdout.jsonl
    "$stage"/zero_shot/inputs/*.jsonl
    "$stage"/inputs/{sciq,piqa}.jsonl
    "$feedback"/generation_followup/data/mmlu_pro.parquet
)
exclude=()
for file in "${reserved[@]}"; do test -s "$file"; exclude+=(--reserve "$file"); done
"$py" prepare_external.py --model "$model" --csqa "$prior_transfer/inputs/csqa_train.jsonl" \
    --obqa "$prior_transfer/inputs/obqa_train.parquet" --gsm8k "$gen/inputs/gsm8k_train.parquet" \
    --science inputs --out data "${exclude[@]}"
"$py" preflight_external.py --data data --model "$model" --config configs/study.json
