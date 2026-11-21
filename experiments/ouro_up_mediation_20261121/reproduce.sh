#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo "Usage: reproduce.sh PYTHON MODEL OUTPUT" >&2; exit 2; }
p="$(realpath "$1")"
h="$(realpath "$2")"
mkdir -p "$3"
r="$(realpath "$3")"
cd "$(dirname "$0")"
export PYTHONPATH="$(realpath ../ouro_native_state_computation_20261121):$(realpath ../ouro_route_content_20261121)"
export OPENBLAS_NUM_THREADS=4
cp -n results/MANIFEST.json results/BODY_MANIFEST.json results/ANGLE_MANIFEST.json "$r/"
for f in openbookqa arc_challenge; do
    "$p" up_mediation.py "$h" "$r/MANIFEST.json" "$r/UP_$f.json" "$f"
    "$p" decode.py "$h" "$r/MANIFEST.json" "$r/DECODE_$f.json" "$f"
    "$p" up_support.py "$h" "$r/MANIFEST.json" "$r/SUPPORT_$f.json" "$f"
    "$p" context_patch.py "$h" "$r/BODY_MANIFEST.json" "$r/BODY_$f.json" "$f"
    "$p" angle.py "$h" "$r/ANGLE_MANIFEST.json" "$r/ANGLE_$f.json" "$f"
done
"$p" analysis_up.py "$r"
"$p" analysis_extra.py "$r"
"$p" figures.py "$r"
"$p" report.py "$r" "$r/REPORT_ZH.md"
"$p" angle_analysis.py "$r"
"$p" angle_report.py "$r" "$r/REPORT_ZH.md"
