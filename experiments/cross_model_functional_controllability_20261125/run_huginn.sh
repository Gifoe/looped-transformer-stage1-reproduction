#!/usr/bin/env bash
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
joint=${OURO_EXTERNAL_ROOT:-/root/ouro-external-data-joint-training-20261124}
model_root=${OURO_TRANSFER_MODELS:-/dev/shm/ouro-cross-model-20261125}
ident=${OURO_IDENTIFICATION_ROOT:-/root/ouro-loop-aware-functional-identification-20261125}
cd "$joint"
source external_env.sh
cd "$study"
export PYTHONPATH="$study/huginn_deps:$study:$ident:$previous:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
test -s "$model_root/huginn-0125/MODEL_LOCK.json"
"$py" -c 'import transformers; assert transformers.__version__ == "4.49.0"'

mkdir -p logs results/huginn-0125
exec 8>"$gpu_lock"
flock 8
"$py" -u huginn_control.py --model "$model_root/huginn-0125" --archive "$archive" --corpus "$corpus" \
    --out results/huginn-0125/fit > logs/fit_huginn-0125.log 2>&1
"$py" -u evaluate_huginn.py --model "$model_root/huginn-0125" \
    --controller results/huginn-0125/fit/rank1_huginn.pt --stage2 "$joint/mixed_full/archive" \
    --followup "$feedback/generation_followup" --reference "$joint/mixed_full/results/FREEZE.json" \
    --out results/huginn-0125 > logs/score_huginn-0125.log 2>&1
"$py" -u audit_huginn_mechanism.py --model "$model_root/huginn-0125" \
    --weights results/huginn-0125/fit/rank1_huginn.pt --archive "$archive" \
    --train "$prior_transfer/data/arc_train.jsonl" --validation "$prior_transfer/data/arc_challenge_validation.jsonl" \
    --out results/huginn-0125/mechanism > logs/mechanism_huginn-0125.log 2>&1
