#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: bash reproduce.sh PYTHON PINNED_MODEL_DIRECTORY OUTPUT_DIRECTORY" >&2
    exit 2
fi
py="$(realpath "$1")"
model="$(realpath "$2")"
mkdir -p "$3"
out="$(realpath "$3")"
cd "$(dirname "$0")"
logs="$out/run_logs"
mkdir -p "$logs"
export PYTHONPATH="$(realpath ../ouro_latent_state_mechanism_v1)${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=4

run() {
    local output="$1" stage="$2"
    shift 2
    if [[ -s "$out/$output" ]]; then
        return
    fi
    "$py" -u "$@" > "$logs/$stage.log" 2>&1 || {
        tail -n 20 "$logs/$stage.log" >&2
        return 1
    }
    test -s "$out/$output"
}

run MANIFEST.json manifest causal_worlds.py "$model" "$out"
run LOCAL.json audit pilot.py "$model" "$out"
run PREDICTION_SHA_pilot.json pilot_prediction predictor.py "$model" "$out" pilot --limit 2 --methods full taylor prefix_anderson
run NATIVE_pilot.json pilot_native evaluator.py "$model" "$out" pilot
sha256sum PREREGISTRATION.md "$out/MANIFEST.json" > "$out/DEVELOPMENT_LOCK.sha256"
run PREDICTION_SHA_development.json development_prediction predictor.py "$model" "$out" development --limit 8
run NATIVE_development.json development_native evaluator.py "$model" "$out" development
run LOCAL_PREDICTION_SHA_development.json development_local_prediction interventions.py "$model" "$out" predict development
run LOCAL_NATIVE_development.json development_local_native interventions.py "$model" "$out" truth development
run LOCAL_FREEZE.json local_lock interventions.py "$model" "$out" freeze
run FREE_METHOD.json method_lock analyze.py "$out" freeze
mapfile -t methods < <("$py" -c 'import json,sys; z=json.load(open(sys.argv[1])); assert z["phase"]=="validation"; print("\n".join(z["methods"]))' "$out/FREE_METHOD.json")
test "${#methods[@]}" -gt 0
sha256sum pilot.py pair_fields.py predictor.py evaluator.py interventions.py natural_and_bridge.py bf16_replay.py analyze.py segments_controls.py PREREGISTRATION.md "$out/MANIFEST.json" "$out/FREE_METHOD.json" "$out/LOCAL_FREEZE.json" > "$out/CONFIRMATION_LOCK.sha256"
run PREDICTION_SHA_confirmation.json confirmation_prediction predictor.py "$model" "$out" confirmation --methods persistence
run NATIVE_confirmation.json confirmation_native evaluator.py "$model" "$out" confirmation
run NATIVE_confirmation_BF16.json confirmation_native_bf16 evaluator.py "$model" "$out" confirmation --bf16
run CONTROLS.json matching_controls segments_controls.py "$model" "$out" controls confirmation
run BRIDGE.json bridge natural_and_bridge.py "$model" "$out" bridge
run LOCAL_PREDICTION_SHA_validation.json validation_local_prediction interventions.py "$model" "$out" predict validation
run LOCAL_NATIVE_validation.json validation_local_native interventions.py "$model" "$out" truth validation
run CLOCKS_AND_BETA.json clocks interventions.py "$model" "$out" clocks
run SHORT.json short segments_controls.py "$model" "$out" short
run PREDICTION_SHA_validation.json validation_prediction predictor.py "$model" "$out" validation --methods "${methods[@]}"
run NATIVE_validation.json validation_native evaluator.py "$model" "$out" validation
run ARC_PREDICTION_SHA.json arc_prediction natural_and_bridge.py "$model" "$out" arc_predict
run ARC_NATIVE.json arc_native natural_and_bridge.py "$model" "$out" arc_truth
run BF16_PREDICTION_SHA.json bf16_replay_prediction bf16_replay.py "$model" "$out" predict
run BF16_NATIVE.json bf16_replay_native bf16_replay.py "$model" "$out" truth
run STATE_AUDIT.json state_audit state_audit.py "$model" "$out"
run ARC_PERMUTATION.json arc_permutation arc_permutation.py "$model" "$out"
"$py" figures.py "$out"
"$py" report_data.py "$out"
"$py" make_report.py "$out" "$out/REPORT_ZH.md"
"$py" verify_release.py "$out" > "$out/RELEASE_VERIFICATION.json"

