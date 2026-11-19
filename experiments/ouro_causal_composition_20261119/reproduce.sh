#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
past=$(realpath "${1:?corrected organization groups.jsonl}")
checkpoint=$(realpath "${2:?fixed model snapshot}")
out=$(realpath -m "${3:?experiment output directory}")
expanded=$(realpath "${4:?expanded experiment output directory}")
stage=${5:-all}
py=${OURO_PYTHON:-python}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} TOKENIZERS_PARALLELISM=false
tasks=(relation logic arithmetic)
wait_all() {
    local rc=0
    for pid in "$@"; do wait "$pid" || rc=1; done
    return "$rc"
}
capture_phase() {
    local phase=$1 inputs=$2
    local running=()
    for task in "${tasks[@]}"; do
        "$py" -u "$here/capture_composition.py" "$inputs" "$out/$phase/$task" \
            "$checkpoint" "$task" "$phase" >"$out/logs/$phase-$task.log" 2>&1 &
        running+=("$!")
    done
    wait_all "${running[@]}"
}
finish() {
    local running=()
    for task in "${tasks[@]}"; do
        test -s "$out/development/$task/completion.json"
        "$py" -u "$here/model_composition.py" "$out/development/$task" "$out/frozen/$task" \
            >"$out/logs/freeze-$task.log" 2>&1 &
        running+=("$!")
    done
    wait_all "${running[@]}"
    (
        cd "$out"
        sha256sum frozen/*/freeze.json data/generation.json data/fresh_groups.jsonl
    ) >"$out/final_freeze.sha256"
    (
        cd "$here"
        sha256sum PROTOCOL.md capture_composition.py independent_groups.py encoders.py \
            model_composition.py evaluate_composition.py summarize_composition.py
    ) >"$out/final_implementation.sha256"
    date -u +%FT%TZ >"$out/independent_started_utc.txt"
    capture_phase independent "$out/data/fresh_groups.jsonl"
    running=()
    for task in "${tasks[@]}"; do
        "$py" -u "$here/evaluate_composition.py" "$out/development/$task" "$out/independent/$task" \
            "$out/frozen/$task" "$checkpoint" "$out/evaluation/$task" >"$out/logs/evaluation-$task.log" 2>&1 &
        running+=("$!")
    done
    wait_all "${running[@]}"
    "$py" -u "$here/summarize_composition.py" "$out" >"$out/logs/summary.log" 2>&1
}
case "$stage" in
    all)
        mkdir "$out"
        mkdir "$out/logs"
        capture_phase pilot "$past"
        "$py" -u "$here/independent_groups.py" "$checkpoint" "$out/data" "$past" \
            "$expanded/inputs.jsonl" "$expanded/competence/inputs.jsonl" \
            "$expanded/screened/pool.jsonl" >"$out/logs/generated.log" 2>&1
        capture_phase development "$past"
        finish
        "$py" -u "$here/semantic_checks.py" "$out" >"$out/logs/semantics.log" 2>&1
        "$py" "$here/figures.py" "$out"
        ;;
    finish)
        finish
        ;;
    *)
        exit 2
        ;;
esac
