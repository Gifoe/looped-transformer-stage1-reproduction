#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
sha256sum -c --quiet PREREGISTRATION.sha256
mkdir -p results
exec 8>"$prior_root/results/gpu.lock"
flock -n 8
ledger=GPU_LEDGER.tsv
touch "$ledger"
[[ $(tail -n 1 "$ledger") != START$'\t'* ]]
used=$(awk -F '\t' '$1=="END" {sum+=$5} END {print sum+0}' "$ledger")
remaining=$((28800-used))
test "$remaining" -gt 15
phase="$1"
shift
start=$(date +%s)
printf 'START\t%s\t%s\t%s\n' "$phase" "$start" "$remaining" >> "$ledger"
code=0
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=15 "$((remaining-15))" "$py" -u "$@" || code=$?
end=$(date +%s)
printf 'END\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$start" "$end" "$((end-start))" "$code" >> "$ledger"
exit "$code"
