#!/usr/bin/env bash
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
joint=${OURO_EXTERNAL_ROOT:-/root/ouro-external-data-joint-training-20261124}
weights=${OURO_TRANSFER_MODELS:-/dev/shm/ouro-cross-model-20261125}
ident=${OURO_IDENTIFICATION_ROOT:-/root/ouro-loop-aware-functional-identification-20261125}
cd "$joint"
source external_env.sh
cd "$study"
export PYTHONPATH="$study:$ident:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
data="$prior_transfer/data"
t1="$controllability/results/weights/rank1_t1.pt"
tail_root="$ident/results"
mkdir -p logs results/mechanism

exec 8>"$gpu_lock"
flock 8
for name in ouro-2.6b-thinking ouro-2.6b ouro-1.4b-thinking; do
    "$py" -u audit_transfer_mechanism.py measure --archive "$archive" --previous "$previous" \
        --models MODELS.json --name "$name" --model "$weights/$name" --t1 "$t1" \
        --train "$data/arc_train.jsonl" --validation "$data/arc_challenge_validation.jsonl" \
        --out "results/mechanism/$name" > "logs/mechanism_$name.log" 2>&1
done
flock -u 8
"$py" -u audit_transfer_mechanism.py summarize --models MODELS.json --t1 "$t1" \
    --base-train "$tail_root/moments/arc_challenge/candidate/k4" --base-finite "$tail_root/finite/arc_k4" \
    --results results/mechanism --out results/mechanism/MECHANISM.json
