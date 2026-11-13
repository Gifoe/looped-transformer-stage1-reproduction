#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ouro-e6a
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
P=venv/bin/python
M=/root/autodl-tmp/ouro-e4-e5/model
$P -u E6B/E6B_HOOK_OVERHEAD_PROBE.py --model "$M" --inputs E6B/E6_TECHNICAL_TRAIN3.jsonl --out E6B/hook_overhead.json > hook_overhead.log 2>&1
$P -u E6A/E6A_ONLINE_EARLY_EXIT.py --model "$M" --inputs E6B/E6_TECHNICAL_TRAIN3.jsonl --native technical_joint_v3 --out E6A/online_technical3 > online_a_technical.log 2>&1
$P -u E6A/E6A_ONLINE_EARLY_EXIT.py --model "$M" --inputs E6B/E6_SHARED_NATIVE_validation.jsonl --native joint/validation --out E6A/online_validation299 > online_a_validation.log 2>&1
echo RUNTIME_VALIDATION_COMPLETE
