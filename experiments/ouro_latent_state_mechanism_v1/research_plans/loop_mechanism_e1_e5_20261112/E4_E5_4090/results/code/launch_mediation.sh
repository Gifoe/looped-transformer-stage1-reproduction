#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ouro-e4-e5
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
BASE=repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/results
MODE=${1:?technical, pilot or validation required}
EXTRA=()
case "$MODE" in
 technical) NAME=mediation_technical2_v1; PRIOR=pilot64_full_v1; COHORT=train64_selection.json; EXTRA=(--limit 2);;
 pilot) NAME=mediation_pilot64_v1; PRIOR=pilot64_full_v1; COHORT=train64_selection.json;;
 validation) NAME=mediation_validation299_v1; PRIOR=validation299_full_v1; COHORT=validation_selection.json;;
 *) exit 2;;
esac
OUT=$BASE/$NAME
mkdir -p "$OUT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/launch_utc.txt"
COMMAND=(venv/bin/python -u state_mediation_runner.py --inputs inputs/runner_inputs.jsonl --cohort inputs/$COHORT --prior "$BASE/$PRIOR" --output "$OUT" "${EXTRA[@]}")
printf '%q ' "${COMMAND[@]}" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
exec "${COMMAND[@]}"
