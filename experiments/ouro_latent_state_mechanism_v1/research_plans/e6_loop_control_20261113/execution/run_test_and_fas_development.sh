#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ouro-e6a
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY=venv/bin/python
M=/root/autodl-tmp/ouro-e4-e5/model
while ! test -f E6A/online_validation299/COMPLETE.json; do sleep 2; done
echo 'c7c9cdb7e6b197e66ac2276e044822e78544478bac0d467f17899529ee9c891a  E6A/joint5090/E6A_FROZEN_METHOD_SPEC.json' | sha256sum --check
echo 'f2b506a42813bfcadb58d12d7c5128f562f1545b99260b6f5532f2226805aae6  E6B/joint5090/E6B_FROZEN_SPEC.json' | sha256sum --check
sha256sum --check GPU_DEVELOPMENT_SOURCE_FREEZE.sha256
$PY -u E6B/E6B_ONLINE_EARLY_EXIT.py --model "$M" --inputs E6B/E6_SHARED_NATIVE_validation.jsonl --native joint/validation --frozen-spec E6B/joint5090/E6B_FROZEN_SPEC.json --out E6B/online_validation299 > online_b_validation.log 2>&1
$PY -u E6B/E6B_NATURAL_FEATURE_EXTRACTION_v3.py --model "$M" --inputs E6B/E6_SHARED_NATIVE_test.jsonl --out joint/test > joint_test.log 2>&1
$PY -u E6A/evaluate_capture_v2.py --capture joint --labels /root/autodl-tmp/ouro-e4-e5/inputs/labels.json --out E6A/joint5090 --phase test > joint_a_frozen_test.log 2>&1
$PY -u E6B/E6_SHARED_FULL_ANSWER_CAPTURE_v2.py --model "$M" --inputs E6A/FULL_ANSWER_train.jsonl --out full_answer/train > fas_train.log 2>&1
$PY -u E6B/E6_SHARED_FULL_ANSWER_CAPTURE_v2.py --model "$M" --inputs E6A/FULL_ANSWER_validation.jsonl --out full_answer/validation > fas_validation.log 2>&1
echo JOINT_TEST_AND_FULL_ANSWER_DEVELOPMENT_COMPLETE
# Full-answer Test waits for independent A/B transfer freeze; no hidden Test job.
