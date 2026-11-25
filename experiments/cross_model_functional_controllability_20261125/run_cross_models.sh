#!/usr/bin/env bash
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
joint=${OURO_EXTERNAL_ROOT:-/root/ouro-external-data-joint-training-20261124}
models=${OURO_TRANSFER_MODELS:-/dev/shm/ouro-cross-model-20261125}
cd "$joint"
source external_env.sh
cd "$study"
export PYTHONPATH="$study:$PYTHONPATH"
mkdir -p logs results
t1="$controllability/results/weights/rank1_t1.pt"
release="$joint/mixed_full"
names=(ouro-2.6b ouro-1.4b-thinking ouro-2.6b-thinking)

for name in "${names[@]}"; do
    while [[ ! -s "$models/$name/MODEL_LOCK.json" ]]; do
        worker=$(cat "fetch_$name.pid")
        if [[ ! -e "/proc/$worker/cwd" ]]; then
            echo "Checkpoint acquisition stopped: $name" >&2
            exit 3
        fi
        sleep 15
    done
    echo READY "$name" "$(date -Is)"
    (
        exec 8>"$gpu_lock"
        flock 8
        "$py" -u evaluate_transfer.py --models MODELS.json --model "$models/$name" --name "$name" \
            --t1 "$t1" --stage2 "$release/archive" --followup "$feedback/generation_followup" \
            --unified "$unified" --reference "$release/results/FREEZE.json" --out "results/$name" \
            >"logs/score_$name.log" 2>&1
    )
    "$py" -u report_transfer.py --results results --out REPORT_ZH.md
done
"$py" -u report_transfer.py --results results --out REPORT_ZH.md --final
printf 'DONE\n'
