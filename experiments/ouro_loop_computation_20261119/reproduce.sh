#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
checkpoint=$(realpath "${1:?fixed model snapshot}")
public=$(realpath "${2:?ARC and HellaSwag parquet folder}")
org=$(realpath "${3:?corrected parent groups.jsonl}")
gsm=$(realpath "${4:?official GSM8K test.jsonl}")
author=$(realpath "${5:?pinned author analysis/attention.py}")
out=$(realpath -m "${6:?fresh output directory}")
py=${OURO_PYTHON:-python}
mkdir "$out"
mkdir "$out/logs"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
tasks=(relation logic arithmetic)
wait_all() {
    local status=0
    for pid in "$@"; do wait "$pid" || status=1; done
    return "$status"
}
"$py" -u "$here/data_loop.py" "$checkpoint" "$out/data" >"$out/logs/generate.log" 2>&1
for t in "${tasks[@]}"; do
    "$py" -u "$here/trace_loop.py" "$checkpoint" "$out/data/groups.jsonl" "$out/pilot/$t" "$t" pilot >"$out/logs/pilot-$t.log" 2>&1
done
running=()
for t in "${tasks[@]}"; do
    "$py" -u "$here/trace_loop.py" "$checkpoint" "$out/data/groups.jsonl" \
        "$out/development/$t" "$t" development >"$out/logs/development-$t.log" 2>&1 &
    running+=("$!")
done
"$py" -u "$here/public_screen.py" "$checkpoint" "$public" "$gsm" "$org" "$out/public" >"$out/logs/public.log" 2>&1 &
running+=("$!")
wait_all "${running[@]}"
"$py" "$here/readouts_loop.py" "$out" freeze >"$out/logs/freeze.log" 2>&1
date -u +%FT%TZ >"$out/freeze_utc.txt"
sha256sum "$here/summary_loop.py" | awk '{print $1}' >"$out/final_metric_code.sha256"
"$py" -u "$here/route_loop.py" "$checkpoint" "$out/data/groups.jsonl" "$out/freeze.json" \
    "$out/route-pilot" relation pilot >"$out/logs/route-pilot.log" 2>&1
running=()
for t in "${tasks[@]}"; do
    (
        "$py" -u "$here/trace_loop.py" "$checkpoint" "$out/data/groups.jsonl" \
            "$out/independent/$t" "$t" independent >"$out/logs/independent-$t.log" 2>&1
        "$py" -u "$here/route_loop.py" "$checkpoint" "$out/data/groups.jsonl" "$out/freeze.json" \
            "$out/causal/$t" "$t" independent >"$out/logs/causal-$t.log" 2>&1
        "$py" -u "$here/scope_loop.py" "$checkpoint" "$out" "$t" >"$out/logs/scope-$t.log" 2>&1
    ) &
    running+=("$!")
done
"$py" -u "$here/macro_reference.py" "$checkpoint" "$gsm" "$author" "$out/paper_reference" \
    >"$out/logs/paper-reference.log" 2>&1 &
running+=("$!")
wait_all "${running[@]}"
"$py" -u "$here/free_generation.py" "$checkpoint" "$out" >"$out/logs/free-generation.log" 2>&1
"$py" -u "$here/readouts_loop.py" "$out" report >"$out/logs/readouts.log" 2>&1
"$py" -u "$here/summary_loop.py" "$out" >"$out/logs/summary.log" 2>&1
"$py" "$here/compare_protocols.py" "$out"
"$py" "$here/figures_loop.py" "$out"
