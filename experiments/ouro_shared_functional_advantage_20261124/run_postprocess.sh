#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
grep -qx DONE run_confirmation.log
sha256sum -c --quiet CONFIRMATION.sha256
export CUDA_VISIBLE_DEVICES=""
"$py" audit_fits.py --fits results/fits --data data --out results/TRAINING_AUDIT.json
"$py" summarize_evaluation.py --phase-dir results/development --out results/PAIRED_DEVELOPMENT.json
"$py" audit_deployed_directions.py --fit results/MOMENT_FIT --holdout results/MOMENT_HOLDOUT \
    --weights results/fits --paired results/PAIRED_DEVELOPMENT.json --development results/development \
    --historical "historical_rank=$prior_transfer/results/HOLDOUT/csqa/source/s261123137/k234/controller.pt" \
    --historical "historical_retention=$gen/results/FINAL_CONTROLLER.pt" \
    --historical "historical_mixed=$gen/results/multitask_full/controller.pt" \
    --out results/DEPLOYED_DIRECTIONS.json > deployed_directions.log
"$py" summarize_evaluation.py --phase-dir results/confirmation --protocol results/CONFIRMATION_LOCK.json \
    --out results/PAIRED_CONFIRMATION.json
"$py" matched_depth.py --protocol results/CONFIRMATION_LOCK.json --results results/confirmation --out results/MATCHED_DEPTH.json
"$py" render_results.py
echo DONE
