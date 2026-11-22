#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
until [[ $(tail -n 1 refine.log) == DONE ]]; do
    ps -p "${1:?refine PID}" -o stat= | grep -qv Z
    sleep 7
done
sha256sum -c results/TRAIN_CODE_LOCK.sha256 results/SUBSPACE_FIT_LOCK.sha256
"$py" prepare_confirmation.py results
if [[ ! -s results/CONFIRMATION_LOCK.sha256 ]]; then
    sha256sum minimal.py train.py benchmark.py generation.py causal.py subspace.py subspace_intervene.py prepare_confirmation.py results/FREEZE.json \
        "$corpus/test.parquet" "$external" > results/CONFIRMATION_LOCK.sha256
fi
sha256sum -c results/CONFIRMATION_LOCK.sha256
mapfile -t ar < <("$py" -c 'import json; v=json.load(open("results/FREEZE.json"));
for k in v["generation_arms"]:
 a=v["arms"][k]; print(k); print(a["file"]); print(a["dose"])')
best="results/${ar[1]}"
dose="${ar[2]}"
if [[ ! -s results/BENCHMARK.json ]]; then
    bash gpu_budget.sh benchmark benchmark.py "$model" "$archive" "$previous" "$legacy" "$corpus" "$external" results > benchmark.log 2>&1
fi
opts=()
for ((i=0; i<${#ar[@]}; i+=3)); do
    opts+=(--minimal "${ar[i]}=results/${ar[i+1]}" --dose "${ar[i]}=${ar[i+2]}")
done
if [[ ! -s results/generation/GENERATION_RESULTS.jsonl ]]; then
    bash gpu_budget.sh generation generation.py --model "$model" --archive "$archive" --corpus "$corpus" \
        --prior "$previous" --legacy "$legacy" --legacy-freeze "$legacy/results/feedback" --external "$external" \
        --out results/generation "${opts[@]}" > generation.log 2>&1
fi
"$py" subspace.py --phase align --out results/SUBSPACE_ANALYSIS --checkpoint "$best"
if [[ ! -s results/SUBSPACE_ANALYSIS/OPENBOOKQA_GRADIENT_ALIGNMENT.json ]]; then
    bash gpu_budget.sh gradient_external subspace.py --phase external --archive "$archive" --model "$model" \
        --out results/SUBSPACE_ANALYSIS --checkpoint "$best" --freeze results/FREEZE.json --external-data "$external" > gradient_external.log 2>&1
fi
if [[ ! -s results/SUBSPACE_ANALYSIS/SUBSPACE_INTERVENTION.json ]]; then
    bash gpu_budget.sh subspace_intervene subspace.py --phase intervene --archive "$archive" --corpus "$corpus" \
        --model "$model" --out results/SUBSPACE_ANALYSIS --checkpoint "$best" --dose "$dose" --batch 8 > subspace_intervene.log 2>&1
fi
if [[ ! -s results/CAUSAL_ANALYSIS/C_POSITIONS.json ]]; then
    bash gpu_budget.sh causal causal.py both --model "$model" --archive "$archive" --corpus "$corpus" \
        --checkpoint "$best" --dose "$dose" --distill results/distill8_t1.pt --constant results/constant_t1.pt \
        --out results/CAUSAL_ANALYSIS > causal.log 2>&1
fi
printf 'DONE\n'
