#!/usr/bin/env bash
set -euo pipefail
if (( $# != 6 )); then
    printf 'Usage: bash reproduce.sh MODEL ARC_CORPUS OPENBOOKQA_TEST FOLLOWUP_DATA FOLLOWUP_RESULTS OUTPUT\n' >&2
    exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
sibling=$(dirname "$here")
model=$(realpath "$1")
corpus=$(realpath "$2")
external=$(realpath "$3")
content_data=$(realpath "$4")
original_history=$(realpath "$5")
mkdir -p "$6"
out=$(realpath "$6")
py=${PYTHON:-python}
archive="$sibling/ouro_arc_pareto_20261119"
previous="$sibling/ouro_semantic_amplification_arc_20261122"
prior="$sibling/ouro_minimal_functional_feedback_20261123"
published="$sibling/ouro_functional_controllability_20261123"
control="$out/reference"
history="$out/history"
export PYTHONPATH="$here:$here/vendor:$published:$prior:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} OPENBLAS_NUM_THREADS=2
mkdir -p "$control/results/GENERATION_RESULTS" "$history" "$out/results/frozen"
ln -sfn "$published/results/weights" "$control/results/weights"
ln -sfn "$published/results/FREEZE.json" "$control/results/FREEZE.json"
ln -sfn "$published/results/CSQA_TRANSFER" "$control/results/CSQA_TRANSFER"
ln -sfn "$published/BENCHMARK_RESULTS" "$control/results/BENCHMARK_RESULTS"
ln -sfn "$published/GENERATION_RESULTS/GENERATION_RESULTS.jsonl" "$control/results/GENERATION_RESULTS/GENERATION_RESULTS.jsonl"
tar -xzf "$published/GENERATION_RESULTS/RUN_FRAGMENTS.tar.gz" -C "$control/results/GENERATION_RESULTS"
for task in gsm8k mmlu_pro; do
    mkdir -p "$history/$task"
    for name in native fixed adaptive feedback; do
        p="$original_history/$task/$name.jsonl"
        if [[ -s "$p" ]]; then
            ln -sfn "$p" "$history/$task/$name.jsonl"
        else
            gzip -dc "$p.gz" > "$history/$task/$name.jsonl"
        fi
    done
done
(cd "$here" && sha256sum -c --quiet ARTIFACTS.sha256)
ledger="$out/GPU_LEDGER.tsv"
touch "$ledger"
exec 9>"$out/gpu.lock"
flock -n 9
run() {
    local name=$1 script=$2 start end elapsed remaining rc=0
    shift 2
    elapsed=$(awk -F '\t' '$1=="END" {seconds+=$5} END {print seconds+0}' "$ledger")
    remaining=$((28800-elapsed))
    test "$remaining" -gt 15
    [[ $(tail -1 "$ledger") != START$'\t'* ]]
    start=$(date +%s)
    printf 'START\t%s\t%s\t%s\n' "$name" "$start" "$remaining" >> "$ledger"
    timeout --signal=TERM --kill-after=15 "$((remaining-15))" "$py" -u "$here/$script" "$@" || rc=$?
    end=$(date +%s)
    printf 'END\t%s\t%s\t%s\t%s\t%s\n' "$name" "$start" "$end" "$((end-start))" "$rc" >> "$ledger"
    return "$rc"
}
r="$out/results"
new="$r/fit/rank1_generation.pt"
freeze="$r/frozen/PROPOSAL.json"
base=(--archive "$archive" --previous "$previous" --model "$model" --corpus "$corpus" --control "$control")
dev=(--ob-validation "$here/data/ob_validation.parquet" --out "$r/dev")
"$py" "$here/reuse_candidates.py" "$control" "$r/BENCHMARK_RESULTS"
run schedule_dev screen.py dev "${base[@]}" "${dev[@]}"
run fit_rank1 fit_rank1.py "${base[@]}" --out "$r/fit"
run rescreen screen.py dev "${base[@]}" "${dev[@]}" --new-weight "$new"
"$py" "$here/freeze_method.py" "$r/dev" "$freeze"
run native_a native_to_rank1.py --archive "$archive" --previous "$previous" --minimal "$prior" \
    --model "$model" --corpus "$corpus" --rank1 "$control/results/weights/rank1_t1.pt" \
    --rank1-freeze "$control/results/FREEZE.json" --out "$r/A_NATIVE" --train-n 128 --train-batch 6 --batch 12
run content_b experiment_b.py --archive "$archive" --model "$model" --prior-study "$control" \
    --minimal-study "$prior" --out "$r/B_CONTENT_PATH"
run generation_confirm screen.py confirm "${base[@]}" --external "$external" \
    --out "$r/BENCHMARK_RESULTS/OPTION_GENERATION" --freeze "$freeze" --new-weight "$new"
"$py" "$here/analyze_option_changes.py" "$control" "$r/BENCHMARK_RESULTS/OPTION_GENERATION"
for task in gsm8k mmlu_pro; do
    run "content_$task" content_eval.py --task "$task" --archive "$archive" --previous "$previous" --model "$model" \
        --control "$control" --data "$content_data" --history "$history" --freeze "$freeze" \
        --new-weight "$new" --out "$r/BENCHMARK_RESULTS/$task"
done
"$py" "$here/accept.py" "$r"
