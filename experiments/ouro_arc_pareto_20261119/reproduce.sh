#!/usr/bin/env bash
set -euo pipefail
if (( $# != 3 )); then
  echo "Usage: $0 /absolute/path/to/model /absolute/path/to/arc-data /absolute/path/to/results" >&2
  exit 2
fi
dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model="$1"
data="$2"
out="$3"
py="${PYTHON:-python}"
mkdir -p "$data" "$out"
revision=fdb98c79578d61bdf502e2ade3253c71154dd11f
for split in validation test; do
  if [[ ! -s "$data/$split.parquet" ]]; then
    curl --fail --location --retry 3 "https://huggingface.co/datasets/allenai/ai2_arc/resolve/$revision/ARC-Challenge/ai2_arc-$split.parquet" -o "$data/$split.parquet.tmp"
    mv "$data/$split.parquet.tmp" "$data/$split.parquet"
  fi
done
(
  cd "$data"
  sha256sum -c <<'EOF'
a0796ca9c64d4633b595dae2b350485de9cc4f45c4adf9b59b371268f945984d  validation.parquet
b8e71916ead552c4e3bb05e3506bfbe4b5caa93b155e680fabb74dcb1ba14ec0  test.parquet
EOF
)
for stage in validation oracle test; do
  if [[ "$stage" == test ]]; then
    "$py" "$dir/analyze_arc.py" freeze "$out"
  fi
  "$py" -u "$dir/run_arc.py" --model "$model" --data "$data" --out "$out" --stage "$stage" --batch-size 16
done
"$py" "$dir/cached_latency.py" "$model" "$data" "$out"
"$py" "$dir/analyze_arc.py" analyze "$out"
"$py" "$dir/oracle_followup.py" "$model" "$data" "$out"
"$py" "$dir/standalone_cost.py" "$model" "$data" "$out"
"$py" "$dir/release_report.py" "$out" "$(dirname "$out")"
