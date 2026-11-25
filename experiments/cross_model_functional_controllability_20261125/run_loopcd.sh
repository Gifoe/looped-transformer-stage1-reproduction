#!/usr/bin/env bash
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
joint=${OURO_EXTERNAL_ROOT:-/root/ouro-external-data-joint-training-20261124}
weights=${OURO_TRANSFER_MODELS:-/dev/shm/ouro-cross-model-20261125}
cd "$joint"
source external_env.sh
cd "$study"
export PYTHONPATH="$study:$PYTHONPATH"
out="$study/results/loopcd"
mkdir -p "$out" logs
exec 8>"$gpu_lock"
flock 8
for name in huginn-0125 ouro-1.4b-thinking ouro-2.6b ouro-2.6b-thinking; do
    dependencies="$PYTHONPATH"
    if [[ "$name" == huginn-0125 ]]; then dependencies="$study/huginn_deps:$dependencies"; fi
    printf 'LOOPCD START %s %s\n' "$name" "$(date -Is)"
    PYTHONPATH="$dependencies" "$py" -u evaluate_loopcd.py \
        --models MODELS.json --name "$name" --model "$weights/$name" \
        --t1 "$controllability/results/weights/rank1_t1.pt" --reference "$joint/mixed_full/results/FREEZE.json" \
        --paper LOOPCD_PAPER.json --transfer "results/$name" --stage2 "$joint/mixed_full/archive" \
        --followup "$feedback/generation_followup" --out "$out/$name" > "logs/loopcd_$name.log" 2>&1
done
