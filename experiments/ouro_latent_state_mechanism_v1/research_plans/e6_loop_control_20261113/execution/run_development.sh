#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ouro-e6a
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY=venv/bin/python
MODEL=/root/autodl-tmp/ouro-e4-e5/model
test -f ENVIRONMENT_READY.json
test -f technical_joint_v3/COMPLETE.json
sha256sum E6B/E6B_NATURAL_FEATURE_EXTRACTION_v3.py E6B/E6_SHARED_FULL_ANSWER_CAPTURE_v2.py > GPU_DEVELOPMENT_SOURCE_FREEZE.sha256
$PY -u E6B/E6B_NATURAL_FEATURE_EXTRACTION_v3.py --model "$MODEL" --inputs E6B/LONG_TRAIN_JOINT_INPUT.jsonl --out long_joint_v3 --technical > long_joint_v3.log 2>&1
$PY -u E6B/E6_SHARED_FULL_ANSWER_CAPTURE_v2.py --model "$MODEL" --inputs E6B/FULL_ANSWER_TECHNICAL_TRAIN3.jsonl --out technical_fas_v2 --technical > technical_fas_v2.log 2>&1
$PY -u E6B/E6_SHARED_FULL_ANSWER_CAPTURE_v2.py --model "$MODEL" --inputs E6B/LONG_TRAIN_FAS_INPUT.jsonl --out long_fas_v2 --technical > long_fas_v2.log 2>&1
sha256sum --check GPU_DEVELOPMENT_SOURCE_FREEZE.sha256
$PY -u E6B/E6B_NATURAL_FEATURE_EXTRACTION_v3.py --model "$MODEL" --inputs E6B/E6_SHARED_NATIVE_train.jsonl --out joint/train > joint_train.log 2>&1
$PY -u E6B/E6B_NATURAL_FEATURE_EXTRACTION_v3.py --model "$MODEL" --inputs E6B/E6_SHARED_NATIVE_validation.jsonl --out joint/validation > joint_validation.log 2>&1
echo DEVELOPMENT_CAPTURE_COMPLETE
# Test is intentionally absent: A/B development freeze must precede its launch.
