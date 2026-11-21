#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo "Usage: reproduce.sh PYTHON PINNED_MODEL OUTPUT" >&2; exit 2; }
py="$(realpath "$1")"
model="$(realpath "$2")"
mkdir -p "$3"
out="$(realpath "$3")"
cd "$(dirname "$0")"
export PYTHONPATH="$(realpath ../ouro_native_state_computation_20261121)"
export OPENBLAS_NUM_THREADS=4
for f in NATURAL_MANIFEST STRUCTURE_MANIFEST PREDICTIONS; do
    cp -n "results/$f.json" "$out/$f.json"
done
for f in openbookqa arc_challenge; do
    "$py" natural.py "$model" "$out/NATURAL_MANIFEST.json" "$out/NATIVE_$f.json" "$f"
    "$py" factorial_native.py "$model" "$out/NATURAL_MANIFEST.json" "$out/FACTORIAL_$f.json" "$f"
    "$py" writer_support.py "$model" "$out/NATURAL_MANIFEST.json" "$out/WRITER_$f.json" "$f"
    "$py" bilinear.py "$model" "$out/NATURAL_MANIFEST.json" "$out/BILINEAR_$f.json" "$f"
done
for phase in pilot confirmation; do
    "$py" structured.py "$model" "$out/STRUCTURE_MANIFEST.json" "$out/PREDICTIONS.json" "$out/STRUCTURE_$phase.json" "$phase"
done
"$py" analysis.py "$out"
"$py" margin_matching.py "$out"
"$py" mechanism_analysis.py "$out"
"$py" bilinear_analysis.py "$out"
"$py" figures.py "$out"
"$py" report.py "$out" "$out/REPORT_ZH.md"

