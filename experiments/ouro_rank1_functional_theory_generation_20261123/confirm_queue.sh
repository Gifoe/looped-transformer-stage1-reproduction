#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
test "$(tail -1 fit_queue.log)" = DONE
mkdir -p results/frozen
freeze=results/frozen/PROPOSAL.json
new=results/fit/rank1_generation.pt
if [[ ! -e "$freeze" ]]; then
    "$py" freeze_method.py results/dev "$freeze"
    cp results/dev/DEV_RESULTS.json results/frozen/DEV_RESULTS.json
    cp results/dev/DEV_LOCK.json results/frozen/DEV_LOCK.json
    date -Iseconds > results/frozen/LOCK_TIME.txt
    sha256sum "$freeze" results/frozen/DEV_RESULTS.json freeze_method.py schedule.py screen.py content_eval.py fit_rank1.py > results/frozen/METHOD.sha256
fi
sha256sum -c --quiet results/frozen/METHOD.sha256
run() {
    local phase=$1
    shift
    if awk -F '\t' -v phase="$phase" '$1=="END" && $2==phase && $6==0 {done=1} END {exit !done}' GPU_LEDGER.tsv; then
        echo "REUSE $phase"
        return
    fi
    echo "RUN $phase"
    bash control_budget.sh "$phase" "$@" > "$phase.log" 2>&1
    echo "COMPLETE $phase"
}
run native_a native_to_rank1.py --archive "$archive" --previous "$previous" --minimal "$prior_root" \
    --model "$model" --corpus "$corpus" --rank1 "$control/results/weights/rank1_t1.pt" \
    --rank1-freeze "$control/results/FREEZE.json" --out results/A_NATIVE --train-n 128 --train-batch 6 --batch 12
run content_b experiment_b.py --model "$model" --archive "$archive" --prior-study "$control" \
    --minimal-study "$prior_root" --out results/B_CONTENT_PATH
run generation_confirm screen.py confirm --archive "$archive" --previous "$previous" --model "$model" \
    --corpus "$corpus" --control "$control" --external "$external" --out results/BENCHMARK_RESULTS/OPTION_GENERATION \
    --freeze "$freeze" --new-weight "$new"
"$py" analyze_option_changes.py "$control" results/BENCHMARK_RESULTS/OPTION_GENERATION
for task in gsm8k mmlu_pro; do
    run "content_$task" content_eval.py --task "$task" --archive "$archive" --previous "$previous" --model "$model" \
        --control "$control" --data "$followup/data" --history "$followup/results" --freeze "$freeze" \
        --new-weight "$new" --out "results/BENCHMARK_RESULTS/$task"
done
printf 'DONE\n'
