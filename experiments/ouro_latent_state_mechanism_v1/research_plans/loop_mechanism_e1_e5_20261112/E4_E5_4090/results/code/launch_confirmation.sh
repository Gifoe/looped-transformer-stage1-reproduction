#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ouro-e4-e5
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
BASE=repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/results
MODE=${1:?technical or full required}
STAGE=${2:?base, query, r2, r3 or r4 required}
EXTRA=()
case "$MODE" in
 technical) SUFFIX=technical2_v1; EXTRA=(--limit 2);;
 full) SUFFIX=full128_v1;;
 *) exit 2;;
esac
PRIOR=$BASE/confirmation_base_$SUFFIX
OUT=$BASE/confirmation_${STAGE}_$SUFFIX
case "$STAGE" in
 base) SCRIPT=confirmation_base_runner.py; ARGS=();;
 query) SCRIPT=confirmation_query_factorial_runner.py; ARGS=(--prior "$PRIOR"); [[ "$MODE" != technical ]] || ARGS+=(--technical);;
 r2|r3|r4) SCRIPT=confirmation_state_mediation_runner.py; ARGS=(--prior "$PRIOR" --receiver "${STAGE#r}");;
 *) exit 2;;
esac
mkdir -p "$OUT"
if [[ -f "$OUT/launch_utc.txt" ]]; then date -u +%Y-%m-%dT%H:%M:%SZ >> "$OUT/resume_launches_utc.txt"; else date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/launch_utc.txt"; fi
COMMAND=(venv/bin/python -u "$SCRIPT" --inputs inputs/confirmation128/inputs.jsonl --cohort inputs/confirmation128/cohort.json --output "$OUT" "${ARGS[@]}" "${EXTRA[@]}")
if [[ -f "$OUT/command.txt" ]]; then COMMAND_FILE=$OUT/resume_commands.txt; else COMMAND_FILE=$OUT/command.txt; fi
printf '%q ' "${COMMAND[@]}" >> "$COMMAND_FILE"
printf '\n' >> "$COMMAND_FILE"
exec "${COMMAND[@]}"
