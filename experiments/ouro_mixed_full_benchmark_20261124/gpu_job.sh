#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source external_env.sh
exec 8>"$gpu_lock"
flock -n 8
label=$1
shift
start=$(date +%s)
printf 'START\t%s\t%s\n' "$label" "$start" >> GPU_LEDGER.tsv
rc=0
CUDA_VISIBLE_DEVICES=0 "$py" -u "$@" || rc=$?
end=$(date +%s)
printf 'END\t%s\t%s\t%s\t%s\t%s\n' "$label" "$start" "$end" "$((end-start))" "$rc" >> GPU_LEDGER.tsv
exit "$rc"
