#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
mkdir -p results
ledger=results/GPU_LEDGER.tsv
exec 8>results/gpu.lock
flock -n 8
touch "$ledger"
[[ $(tail -n 1 "$ledger") != START$'\t'* ]]
used=$(awk -F '\t' '$1=="END" {sum+=$5} END {print sum+0}' "$ledger")
left=$((18000-used))
test "$left" -gt 15
phase="$1"
shift
start=$(date +%s)
printf 'START\t%s\t%s\t%s\n' "$phase" "$start" "$left" >> "$ledger"
set +e
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=15 "$((left-15))" "$py" -u "$@"
code=$?
set -e
end=$(date +%s)
printf 'END\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$start" "$end" "$((end-start))" "$code" >> "$ledger"
exit "$code"
