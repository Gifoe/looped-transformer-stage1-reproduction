#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-causal-semantic-completion-20261122
source env.sh
run=results/GPU_LEDGER.tsv
start="$(date +%s)"
limit=18000
used="$(awk -F '\t' '$1=="END" {sum+=$5} END {print sum+0}' "$run" 2>/dev/null || true)"
last="$(tail -n 1 "$run" 2>/dev/null || true)"
if [[ "$last" == START$'\t'* ]]; then
    echo "Unfinished GPU stage in ledger"
    exit 1
fi
left="$((limit-${used:-0}))"
test "$left" -gt 0
stamp="$1"
shift
exec 8>results/gpu.lock
flock -n 8
printf 'START\t%s\t%s\t%s\n' "$stamp" "$start" "$left" >> "$run"
set +e
timeout --signal=TERM --kill-after=15 "$((left>15?left-15:left))" "$py" -u "$@"
code=$?
set -e
end="$(date +%s)"
printf 'END\t%s\t%s\t%s\t%s\t%s\n' "$stamp" "$start" "$end" "$((end-start))" "$code" >> "$run"
exit "$code"
