#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source external_env.sh
export feedback advantage model unified
"$py" mixed_full/prepare_mixed.py
out=mixed_full/results
args=(--archive "$archive" --previous "$previous" --model "$model" --minimal "$feedback" --stage2 mixed_full/archive --unified "$unified" --followup "$feedback/generation_followup" --historical "$historical" --out "$out/predictions" --freeze "$out/FREEZE.json" --arm mixed)
for task in sciq piqa arc_challenge arc_easy winogrande mmlu hellaswag mmlu_pro; do
    printf '%s START %s\n' "$(date -Is)" "$task"
    bash gpu_job.sh "mixed_full_$task" mixed_full/mixed_zero_entry.py "${args[@]}" --task "$task" >> "mixed_full/logs/$task.log" 2>&1
    printf '%s COMPLETE %s\n' "$(date -Is)" "$task"
done
printf 'DONE\n'
