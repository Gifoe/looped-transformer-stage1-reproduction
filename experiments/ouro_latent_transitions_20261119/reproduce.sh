#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
past=$(realpath "${1:?corrected 20261118 server study output}")
model=$(realpath "${2:?fixed Hugging Face model snapshot}")
out=$(realpath -m "${3:?new experiment output folder}")
py=${OURO_PYTHON:-python}
mkdir "$out"
mkdir "$out"/{runs,logs}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
tasks=(relation logic arithmetic)
wait_jobs() {
    local status=0
    for pid in "$@"; do wait "$pid" || status=1; done
    return "$status"
}
for t in "${tasks[@]}"; do
    "$py" -u "$here/capture_transition.py" "$past" "$out/runs" "$model" "$t" --pilot >"$out/logs/pilot-$t.log" 2>&1
done
pids=()
for t in "${tasks[@]}"; do
    "$py" -u "$here/capture_transition.py" "$past" "$out/runs" "$model" "$t" >"$out/logs/capture-$t.log" 2>&1 &
    pids+=("$!")
done
wait_jobs "${pids[@]}"
pids=()
for t in "${tasks[@]}"; do
    (
        "$py" -u "$here/fit_study.py" "$past" "$out/runs" "$out/results" "$model" "$t" 96 >"$out/logs/eval-96-$t.log" 2>&1
        "$py" -u "$here/fit_study.py" "$past" "$out/runs" "$out/results" "$model" "$t" 2048 >"$out/logs/eval-full-$t.log" 2>&1
    ) &
    pids+=("$!")
done
"$py" -u "$here/case_study.py" "$past" "$out/runs" "$out/full_cases" "$model" >"$out/logs/full-cases.log" 2>&1 &
pids+=("$!")
wait_jobs "${pids[@]}"
"$py" -u "$here/transfer.py" "$past" "$out/runs" "$out/results" >"$out/logs/cross-task.log" 2>&1
"$py" -u "$here/summarize_transition.py" "$out/runs" "$out/results" >"$out/logs/summary.log" 2>&1
"$py" -u "$here/diagnose_shift.py" "$past" "$out/runs" "$out/results" >"$out/logs/shift.log" 2>&1
"$py" -u "$here/novel_inputs.py" "$past" "$out/runs" "$out/results" >"$out/logs/novel.log" 2>&1
"$py" -u "$here/figures.py" "$out/results" "$out/full_cases/cases.json" >"$out/logs/figures.log" 2>&1
