#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 5 )); then
    printf '%s\n' 'usage: reproduce.sh STAGE MODEL_DIR ARC_TRAIN_PARQUET OPENBOOKQA_TRAIN_PARQUET RUN_DIR' >&2
    printf '%s\n' 'STAGE is train-dev, postgate-eval, postgate-cost or all.' >&2
    exit 2
fi
stage="$1"
model="$2"
arc="$3"
openbook="$4"
output="$5"
py="${PYTHON:-python}"
freeze="$here/EXECUTION_FREEZE.json"
manifest="$here/SPLIT_MANIFEST.json"
gate="$output/dev/COMPLETE.json"

dev_passed() {
    "$py" - "$gate" <<'PY'
import json
from pathlib import Path
import sys
gate = Path(sys.argv[1])
sys.exit(not (gate.is_file() and json.loads(gate.read_bytes())["passed"]))
PY
}

wait_for_exclusive_gpu() {
    while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; do
        sleep 10
    done
}

train_dev() {
    test ! -e "$gate"
    mkdir -p "$output"
    "$py" "$here/native_parity.py" "$model" "$arc" "$output/NATIVE_DEPTH6_PARITY.json"
    "$py" -u "$here/train_shared.py" "$model" "$arc" "$manifest" "$output" "$freeze"
    "$py" -u "$here/evaluate_shared.py" "$model" "$arc" "$manifest" "$output" "$output" "$freeze" --cohort dev
}

postgate_eval() {
    dev_passed
    test ! -e "$output/transfer"
    test ! -e "$output/diagnostic"
    mkdir -p "$output/postgate"
    "$py" -u "$here/evaluate_shared.py" "$model" "$openbook" "$manifest" "$output" "$output" "$freeze" --cohort transfer --gate "$gate"
    "$py" -u "$here/evaluate_shared.py" "$model" "$arc" "$manifest" "$output" "$output" "$freeze" --cohort diagnostic --gate "$gate"
    "$py" -u "$here/deployment_shared.py" "$model" "$arc" "$manifest" "$output" "$gate" "$freeze" "$output/postgate" --stage recurrence
}

postgate_cost() {
    dev_passed
    test -s "$output/transfer/COMPLETE.json"
    test -s "$output/diagnostic/COMPLETE.json"
    test -s "$output/postgate/RECURRENT_SWITCH.json"
    test ! -e "$output/postgate/STANDALONE_COST_ROWS.jsonl"
    wait_for_exclusive_gpu
    nvidia-smi --query-compute-apps=pid --format=csv > "$output/postgate/UNCONTENDED_GPU_START.txt"
    "$py" -u "$here/deployment_shared.py" "$model" "$arc" "$manifest" "$output" "$gate" "$freeze" "$output/postgate" --stage cost
    date -u --iso-8601=seconds > "$output/postgate/queue_complete.txt"
}

case "$stage" in
    train-dev) train_dev ;;
    postgate-eval) postgate_eval ;;
    postgate-cost) postgate_cost ;;
    all)
        train_dev
        if dev_passed; then
            postgate_eval
            postgate_cost
        else
            printf '%s\n' 'Development gate closed. Both reserved cohorts remain sealed.'
        fi
        ;;
    *)
        printf 'Unknown stage: %s\n' "$stage" >&2
        exit 2
        ;;
esac

