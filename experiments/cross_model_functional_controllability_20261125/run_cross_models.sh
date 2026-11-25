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
historical_result="$joint/t1_full/results/RESULTS.json"
pending=(ouro-2.6b ouro-1.4b-thinking ouro-2.6b-thinking)

while [[ ${#pending[@]} -gt 0 ]]; do
    name=
    for candidate in "${pending[@]}"; do
        if [[ -s "$models/$candidate/MODEL_LOCK.json" ]]; then
            name=$candidate
            break
        fi
    done
    if [[ -z "$name" ]]; then
        for candidate in "${pending[@]}"; do
            worker=$(cat "fetch_$candidate.pid")
            if [[ ! -e "/proc/$worker/cwd" ]] && [[ ! -s "$models/$candidate/MODEL_LOCK.json" ]]; then
                echo "Checkpoint acquisition stopped: $candidate" >&2
                exit 3
            fi
        done
        sleep 15
        continue
    fi
    echo READY "$name" "$(date -Is)"
    (
        exec 8>"$gpu_lock"
        flock 8
        "$py" -u evaluate_transfer.py --models MODELS.json --model "$models/$name" --name "$name" \
            --t1 "$t1" --stage2 "$release/archive" --followup "$feedback/generation_followup" \
            --unified "$unified" --reference "$release/results/FREEZE.json" --out "results/$name" \
            >"logs/score_$name.log" 2>&1
    )
    "$py" -u report_transfer.py --results results --out REPORT_ZH.md --historical "$historical_result"
    next=()
    for candidate in "${pending[@]}"; do
        if [[ "$candidate" != "$name" ]]; then next+=("$candidate"); fi
    done
    pending=("${next[@]}")
done
"$py" -u report_transfer.py --results results --out REPORT_ZH.md --historical "$historical_result" --final
"$py" -u package_transfer.py --root "$study" --models "$models" --t1 "$t1" --historical "$historical_result" \
    --out /root/ouro-cross-model-functional-controllability-release-20261125
printf 'DONE\n'
