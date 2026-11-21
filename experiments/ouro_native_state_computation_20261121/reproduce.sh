#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo "Usage: reproduce.sh PYTHON MODEL OUTPUT" >&2; exit 2; }
py="$(realpath "$1")"
model="$(realpath "$2")"
mkdir -p "$3"
out="$(realpath "$3")"
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=4
mkdir -p "$out/logs"
run() {
    local receipt="$1" stage="$2"
    shift 2
    if [ ! -s "$out/$receipt" ]; then
        "$py" -u "$@" > "$out/logs/$stage.log" 2>&1
    fi
}
if [[ -f smoke.json ]]; then
    run RUNTIME.json technical observe.py "$model" "$out" smoke.json technical
else
    run RUNTIME.json technical observe.py "$model" "$out" ARC_BLIND_INPUTS.json technical
fi
for phase in pilot dev; do
    run "ATLAS_$phase.json" "atlas_$phase" observe.py "$model" "$out" ARC_BLIND_INPUTS.json "$phase"
done
run CANDIDATES.json candidates observe.py "$model" "$out" ARC_BLIND_INPUTS.json freeze
# Incrementally written behavior and causal logs can be resumed with the same command.
"$py" causal.py "$model" "$out" BLIND.json dev --limit 48
"$py" behavior.py "$model" "$out" INPUTS.json dev
run GATE.json gate gate.py "$out"
run PREDICTIONS.json freeze causal.py "$model" "$out" BLIND.json predict
sha256sum native.py observe.py data_protocol.py behavior.py causal.py gate.py origin.py semantic.py full_tf.py PREREGISTRATION.md BLIND.json \
    "$out/CANDIDATES.json" "$out/GATE.json" "$out/PREDICTIONS.json" > "$out/CONFIRMATION_LOCK.sha256"
for phase in confirm_arc confirm_external controlled; do
    run "ATLAS_$phase.json" "atlas_$phase" observe.py "$model" "$out" BLIND.json "$phase"
    "$py" causal.py "$model" "$out" BLIND.json "$phase"
    "$py" behavior.py "$model" "$out" INPUTS.json "$phase"
done
for phase in confirm_arc confirm_external; do
    run "FULL_PATCH_$phase.json" "full_$phase" full_tf.py "$model" "$out" BLIND.json "$phase"
done
run SEMANTIC_NATIVE.json semantics semantic.py "$model" "$out" STRUCTURE.json
"$py" origin_formation.py "$model" "$out" BLIND.json FORMATION_PROTOCOL.md
run RESPONSE_CONTROLS.json response response_controls.py "$model" "$out" BLIND.json
"$py" resources.py "$out"
"$py" summarize.py "$out" INPUTS.json STRUCTURE.json
"$py" figures.py "$out"
"$py" documents.py "$out" "$out/docs"

