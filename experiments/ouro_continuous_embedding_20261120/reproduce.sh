#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
    printf 'usage: %s PINNED_MODEL_DIR RESULTS_DIR\n' "$0" >&2
    exit 2
fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
out="${2%/}"
py="${PYTHON:-python}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
mkdir -p "$out/logs"
for stage in pilot operators reconstruct; do
    "$py" -u "$here/embedding_run.py" "$1" "$out" "$stage" 2>&1 | tee "$out/logs/$stage.log"
done
"$py" "$here/freeze_confirmation.py" "$1" "$out"
for stage in holdout causal; do
    "$py" -u "$here/embedding_run.py" "$1" "$out" "$stage" 2>&1 | tee "$out/logs/$stage.log"
done
"$py" -u "$here/operator_confirmation.py" "$1" "$out" 2>&1 | tee "$out/logs/operator_confirmation.log"
"$py" -u "$here/prefix_control.py" "$1" "$out" 2>&1 | tee "$out/logs/prefix_control.log"
"$py" "$here/figures.py" "$out"
if [[ "$(basename "$out")" == results ]]; then
    "$py" "$here/report.py" "$out" "$(dirname "$out")"
fi
