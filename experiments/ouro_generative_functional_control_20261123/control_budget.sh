#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
sha256sum -c --quiet PREREGISTRATION.sha256
exec 8>"$gpu_lock"
flock -n 8
touch GPU_LEDGER.tsv
[[ "$(tail -n 1 GPU_LEDGER.tsv)" != START$'\t'* ]]
used=$(awk -F '\t' '$1=="END"{s+=$5} END{print s+0}' GPU_LEDGER.tsv)
remaining=$((18000-used))
test "$remaining" -gt 15
phase=$1
shift
now=$(date +%s)
printf 'START\t%s\t%s\t%s\n' "$phase" "$now" "$remaining" >> GPU_LEDGER.tsv
rc=0
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=15 "$((remaining-15))" "$py" -u "$@" || rc=$?
end=$(date +%s)
printf 'END\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$now" "$end" "$((end-now))" "$rc" >> GPU_LEDGER.tsv
exit "$rc"
