#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
v=/tmp/ouro-stage2-requests-cpu-20261123/venv/bin/python
export OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
gpu="$1"
prep="$2"
while [[ $(tail -n 1 prepare_grade.log) != DONE ]]; do
    ps -p "$prep" -o stat= | grep -qv Z
    sleep 15
done
for task in humaneval mbpp; do
    until [[ -s results/$task/feedback.jsonl ]]; do
        ps -p "$gpu" -o stat= | grep -qv Z
        sleep 15
    done
    if [[ $(find results/$task -maxdepth 1 -name "*.evalplus.jsonl" | wc -l) != 4 ]]; then
        "$v" -u generation_followup_evalplus.py --jail eval_jail --results results --task "$task" > "grade_$task.log" 2>&1
    fi
    echo "COMPLETE $task"
done
printf "DONE\n"
