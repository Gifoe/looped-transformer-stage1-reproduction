#!/usr/bin/env bash
set -euo pipefail
if (( $# != 4 )); then
  echo 'Usage: reproduce.sh train|frozen MODEL_DIR ARC_PARQUET_DIR OUTPUT_DIR' >&2
  exit 2
fi
mode="$1"
[[ "$mode" == train || "$mode" == frozen ]]
dir="$(cd "$(dirname "$0")" && pwd)"
archive="$dir/../ouro_arc_pareto_20261119"
model="$2"
corpus="$3"
out="$4"
py="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENBLAS_NUM_THREADS=4
mkdir -p "$out" "$corpus"
out="$(realpath "$out")"
test "$out" != "$dir/results"
ledger="$out/GPU_LEDGER.tsv"
test ! -e "$ledger"
revision=fdb98c79578d61bdf502e2ade3253c71154dd11f
for split in train validation test; do
  if [[ ! -s "$corpus/$split.parquet" ]]; then
    curl --fail --location --retry 3 "https://huggingface.co/datasets/allenai/ai2_arc/resolve/$revision/ARC-Challenge/ai2_arc-$split.parquet" -o "$corpus/$split.parquet.tmp"
    mv "$corpus/$split.parquet.tmp" "$corpus/$split.parquet"
  fi
done
"$py" - "$dir/results" "$corpus" <<'PY'
from pathlib import Path
import hashlib,json,sys
r,d=map(Path,sys.argv[1:])
expected=json.loads((r/'TRAIN_INPUT.json').read_text())
expected['test']=json.loads((r/'TEST_INPUT.json').read_text())['dataset_sha256']
for split in ('train','validation','test'):
    with (d/f'{split}.parquet').open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==expected[split], split
PY
remaining=18000
stage() {
  name="$1"
  shift
  test "$remaining" -gt 15
  start=$(date +%s)
  printf 'START\t%s\t%s\t%s\n' "$name" "$start" "$remaining" >> "$ledger"
  code=0
  timeout --signal=TERM --kill-after=15 "$((remaining-15))" "$py" -u "$@" > "$out/$name.log" 2>&1 || code=$?
  end=$(date +%s)
  elapsed=$((end-start))
  printf 'END\t%s\t%s\t%s\t%s\t%s\n' "$name" "$start" "$end" "$elapsed" "$code" >> "$ledger"
  remaining=$((remaining-elapsed))
  if (( code != 0 )); then
    tail -n 25 "$out/$name.log" >&2
    exit "$code"
  fi
}
if [[ "$mode" == train ]]; then
  stage features "$dir/arc_features.py" "$model" "$corpus" "$out/features" "$archive"
  stage training "$dir/amplify.py" "$model" "$out/features" "$out" "$archive"
else
  cp "$dir/results/FREEZE.json" "$dir/results/"*_round*.pt "$out/"
fi
sha256sum "$dir/equal_dose.py" > "$out/GEOMETRY_CODE_LOCK.sha256"
stage test "$dir/evaluate.py" "$model" "$corpus" "$out" "$archive"
stage geometry "$dir/equal_dose.py" "$model" "$corpus" "$out" "$archive"
"$py" "$dir/audit.py" "$out" "$archive" "$corpus"
"$py" "$dir/report.py" "$out" "$out/REPORT_ZH.md"
