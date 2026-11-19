#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 4 )); then
    echo "usage: $0 MODEL_DIRECTORY ARC_PARQUET_DIRECTORY OUTPUT_DIRECTORY PHASE" >&2
    exit 2
fi
model="$1"
arc="$2"
out="$3"
phase="$4"
py="${PYTHON:-python}"
prior="$here/../ouro_arc_pareto_20261119/results"
historical="$here/../ouro_latent_state_mechanism_v1/results/orchestration_runs/resume-test-20261111/behavior_test.json"

case "$phase" in
    native)
        "$py" "$here/protocol_arc.py" --phase plan --model "$model" --data "$arc" --out "$out"
        "$py" -u "$here/protocol_arc.py" --phase native --model "$model" --data "$arc" --out "$out" --archive "$prior" --batch 16 --max-tokens 24576 --run-rotation
        "$py" "$here/analyze_protocol.py" "$out" "$historical"
        "$py" -u "$here/protocol_context_control.py" --model "$model" --data "$arc" --out "$out"
        ;;
    loopcd)
        "$py" -u "$here/protocol_arc.py" --phase loopcd --model "$model" --data "$arc" --out "$out" --archive "$prior" --batch 16 --max-tokens 24576 --conditions A B C D E F
        "$py" "$here/analyze_protocol.py" "$out" "$historical" --loopcd
        ;;
    costs)
        "$py" -u "$here/protocol_cost_generation.py" --model "$model" --data "$arc" --out "$out" --generation-n 16 --max-new 12
        "$py" "$here/protocol_final_audit.py" "$out"
        "$py" "$here/write_protocol_report.py" "$out"
        ;;
    *)
        echo "unknown phase: $phase" >&2
        exit 2
        ;;
esac
