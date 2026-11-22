#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
v=/tmp/ouro-stage2-requests-cpu-20261123/venv/bin/python
export OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
"$v" -u generation_followup_evalplus.py --jail eval_jail --results results --smoke-only
for a in humaneval mbpp; do
    "$v" -u generation_followup_evalplus.py --jail eval_jail --results results --prepare --task "$a"
done
printf "DONE\n"
