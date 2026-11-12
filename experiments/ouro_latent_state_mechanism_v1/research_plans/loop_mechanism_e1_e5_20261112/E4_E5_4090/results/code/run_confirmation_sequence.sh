#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ouro-e4-e5
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
MODE=${1:?technical or full required}
case "$MODE" in technical) SUFFIX=technical2_v1; N=2;; full) SUFFIX=full128_v1; N=128;; *) exit 2;; esac
BASE=repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/results
mkdir -p "$BASE/audits"
venv/bin/python -c 'import json,pathlib; d=json.load(open("inputs/confirmation128/analysis_labels.json")); p=pathlib.Path("confirmation128_analysis_labels.jsonl"); s="".join(json.dumps(v)+"\n" for k,v in sorted(d.items())); p.write_text(s)'
for STAGE in base query r2 r3 r4; do
 OUT=$BASE/confirmation_${STAGE}_$SUFFIX
 bash launch_confirmation.sh "$MODE" "$STAGE" >> "$BASE/confirmation_${STAGE}_$SUFFIX.log" 2>&1
 AUDIT=$BASE/audits/CONFIRMATION_${STAGE}_${SUFFIX}_AUDIT.json
 COMMON=(--results "$OUT" --labels confirmation128_analysis_labels.jsonl --cohort inputs/confirmation128/cohort.json --output "$AUDIT")
 case "$STAGE" in
  base) CUDA_VISIBLE_DEVICES="" venv/bin/python confirmation_base_audit.py "${COMMON[@]}" --inputs inputs/confirmation128/inputs.jsonl --expected-n "$N";;
  query) CUDA_VISIBLE_DEVICES="" venv/bin/python astra_query_audit.py "${COMMON[@]}" --prior "$BASE/confirmation_base_$SUFFIX";;
  r2) CUDA_VISIBLE_DEVICES="" venv/bin/python astra_mediation_audit.py "${COMMON[@]}" --prior "$BASE/confirmation_base_$SUFFIX";;
  r3|r4) CUDA_VISIBLE_DEVICES="" venv/bin/python astra_late_mediation_audit.py "${COMMON[@]}" --prior "$BASE/confirmation_base_$SUFFIX" --receiver "${STAGE#r}";;
 esac
 venv/bin/python verify_confirmation_completion.py "$OUT" inputs/confirmation128/cohort.json "$STAGE" "$N"
 venv/bin/python execution_resource_audit.py "$OUT"
done
