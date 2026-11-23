#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
sha256sum --check --quiet PREREGISTRATION.sha256
exec 8>"$prior_root/results/gpu.lock"
flock -n 8
ledger=GPU_LEDGER.tsv
touch "$ledger"
[[ "$(tail -1 "$ledger")" != START$'\t'* ]]
used=$(awk -F '\t' '$1=="END"{s+=$5} END{print s+0}' "$ledger")
remaining=$((18000-used))
test "$remaining" -gt 15
stage=$1
shift
start=$(date +%s)
printf 'START\t%s\t%s\t%s\n' "$stage" "$start" "$remaining" >> "$ledger"
status=0
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=15 "$((remaining-15))" "$py" -u "$@" || status=$?
end=$(date +%s)
printf 'END\t%s\t%s\t%s\t%s\t%s\n' "$stage" "$start" "$end" "$((end-start))" "$status" >> "$ledger"
exit "$status"
