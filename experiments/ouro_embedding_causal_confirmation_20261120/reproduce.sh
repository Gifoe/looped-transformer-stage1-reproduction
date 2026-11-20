#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
    printf 'usage: %s PINNED_MODEL_DIR RESULTS_DIR\n' "$0" >&2
    exit 2
fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
result="${2%/}"
py="${PYTHON:-python}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
mkdir -p "$result/logs"
"$py" -u "$here/confirm.py" "$1" "$result" freeze 2>&1 | tee "$result/logs/freeze.log"
for dtype in float32 bfloat16; do
    for stage in natural causal; do
        "$py" -u "$here/confirm.py" "$1" "$result" "$stage" --dtype "$dtype" 2>&1 | tee "$result/logs/$stage.$dtype.log"
    done
done
"$py" -u "$here/context_check.py" "$1" "$result" 2>&1 | tee "$result/logs/donor_context.log"
"$py" "$here/render.py" "$result" "$(dirname "$result")"
