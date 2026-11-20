#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "usage: $0 PINNED_MODEL_DIR ARC_CHALLENGE_VALIDATION_PARQUET OUTPUT_DIR" >&2
    exit 2
fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tableau="$here/../ouro_hidden_computation_20261120/results/mechanism/mechanism_tableaux.jsonl"
py="${PYTHON:-python}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
"$py" -u "$here/study.py" "$1" "$tableau" "$2" "$3" pilot
"$py" -u "$here/study.py" "$1" "$tableau" "$2" "$3" formal
"$py" "$here/analyze.py" "$3"
"$py" "$here/audit_runtime.py" "$1" "$3"
"$py" -u "$here/confirmation.py" "$1" "$tableau" "$2" "$3/confirmation"
"$py" "$here/confirmation_charts.py" "$3"
"$py" "$here/report.py" "$3" "$(dirname "$3")"
