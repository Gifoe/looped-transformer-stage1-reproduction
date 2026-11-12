#!/usr/bin/env bash
# Whole ARC dataset K1..K4 acquisition: original joint-option order, no B pilot.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
ROOT="${OURO_OUTPUT_ROOT:-$PWD/outputs}"
mkdir -p "$ROOT"
python verify_runtime.py --input data/validation.jsonl --out "$ROOT/runtime_parity.json"
python storage_plan.py --inputs data/train.jsonl data/validation.jsonl data/test.jsonl \
  --output-mount "$ROOT" --capture current --save-head | tee "$ROOT/SPACE_FORECAST.json"
for split in train validation test; do
  python collect.py --input "data/${split}.jsonl" --out "$ROOT/${split}_native" \
    --capture current --save-head
  extra=()
  if [[ "$split" == "train" ]]; then
    extra=(--input-manifest data/train.jsonl --exclude-fewshot)
  fi
  if [[ "$split" == "test" ]]; then
    extra+=(--expect-full-arc-test)
  fi
  python analyze.py --predictions "$ROOT/${split}_native/predictions.jsonl" \
    --labels "data/${split}.labels.jsonl" --out "$ROOT/analyses/${split}" "${extra[@]}"
done
