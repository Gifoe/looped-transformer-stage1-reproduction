#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
export PYTHONPATH="$PWD:$PYTHONPATH"
p=gen_followup; j=generation_followup
for task in "$@"; do
  echo "RUN $task"
  if ! bash gpu_budget.sh "${p}_$task" "$j/generation_followup_run.py" --archive "$archive" --previous "$previous" --model "$model" --data "$j/data" --out "$j/results" --freeze results/FREEZE.json --task "$task" > "$j/$task.log" 2>&1; then
    echo "FAILED $task"
    tail -16 "$j/$task.log"
    exit 1
  fi
  echo "COMPLETE $task"
done
printf "DONE\n"
