#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ouro-e4-e5
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
BASE=repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/results
MODE=${1:?technical or validation required}
DEPTH=${2:?receiver depth 3 or 4 required}
[[ "$DEPTH" == 3 || "$DEPTH" == 4 ]]
EXTRA=()
case "$MODE" in
 technical) NAME=depth${DEPTH}_mediation_technical2_v1; EXTRA=(--limit 2);;
 validation) NAME=depth${DEPTH}_mediation_validation299_v1;;
 *) exit 2;;
esac
OUT=$BASE/$NAME
mkdir -p "$OUT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/launch_utc.txt"
COMMAND=(venv/bin/python -u depth_state_mediation_runner.py --receiver "$DEPTH" --inputs inputs/runner_inputs.jsonl --cohort inputs/validation_selection.json --prior "$BASE/validation299_full_v1" --output "$OUT" "${EXTRA[@]}")
printf '%q ' "${COMMAND[@]}" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
exec "${COMMAND[@]}"
