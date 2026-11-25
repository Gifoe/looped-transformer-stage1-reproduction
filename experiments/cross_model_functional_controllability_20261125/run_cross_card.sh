#!/usr/bin/env bash
set -euo pipefail
host_mode=$1
physical_gpu=$2
name=$3
study=$(cd "$(dirname "$0")" && pwd)
joint=/root/ouro-external-data-joint-training-20261124
weights=/dev/shm/ouro-cross-model-20261125
ident=/root/ouro-loop-aware-functional-identification-20261125
cd "$joint"
source external_env.sh
cd "$study"
export PYTHONPATH="$study:$ident:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="$physical_gpu"
mkdir -p logs locks results/loopcd results/mechanism
if [[ "$host_mode" == old ]]; then
    exec 8>"$gpu_lock"
else
    exec 8>"$study/locks/card_$physical_gpu.lock"
fi
flock 8
if [[ "$host_mode" == old ]]; then
    "$py" - <<'PY'
import json
from pathlib import Path
p = Path("results/loopcd/huginn-0125/RESULTS.json")
huginn = json.loads(p.read_text())
assert len(huginn) == 8
row = huginn["mmlu_pro"]
assert row["source"] == "current_matched" and set(row["arms"]) == {"fixed", "adaptive"}
PY
else
    while [[ ! -s "$weights/$name/MODEL_LOCK.json" ]]; do
        fetch=$(cat "fetch_$name.remote.pid")
        if ! ps -p "$fetch" -o stat= | grep -q "^[DRSTI]"; then
            printf 'Checkpoint fetch ended before the receipt for %s\n' "$name" >&2
            exit 1
        fi
        sleep 5
    done
fi
printf 'LOOPCD START %s %s host=%s gpu=%s\n' "$name" "$(date -Is)" "$(hostname)" "$physical_gpu"
"$py" -u evaluate_loopcd.py --models MODELS.json --name "$name" --model "$weights/$name" \
    --t1 "$controllability/results/weights/rank1_t1.pt" --reference "$joint/mixed_full/results/FREEZE.json" \
    --paper LOOPCD_PAPER.json --transfer "results/$name" --stage2 "$joint/mixed_full/archive" \
    --followup "$feedback/generation_followup" --out "results/loopcd/$name" >"logs/loopcd_$name.log" 2>&1
printf 'MECHANISM START %s %s\n' "$name" "$(date -Is)"
data="$prior_transfer/data"
"$py" -u audit_transfer_mechanism.py measure --archive "$archive" --previous "$previous" \
    --models MODELS.json --name "$name" --model "$weights/$name" --t1 "$controllability/results/weights/rank1_t1.pt" \
    --train "$data/arc_train.jsonl" --validation "$data/arc_challenge_validation.jsonl" \
    --out "results/mechanism/$name" >"logs/mechanism_$name.log" 2>&1
printf 'DONE %s %s\n' "$name" "$(date -Is)"
