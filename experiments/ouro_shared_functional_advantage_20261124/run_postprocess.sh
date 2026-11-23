#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
grep -qx DONE run_confirmation.log
sha256sum -c --quiet CONFIRMATION.sha256
export CUDA_VISIBLE_DEVICES=""
"$py" audit_fits.py --fits results/fits --data data --out results/TRAINING_AUDIT.json
"$py" summarize_evaluation.py --phase-dir results/development --out results/PAIRED_DEVELOPMENT.json
"$py" summarize_evaluation.py --phase-dir results/confirmation --protocol results/CONFIRMATION_LOCK.json \
    --out results/PAIRED_CONFIRMATION.json
"$py" matched_depth.py --protocol results/CONFIRMATION_LOCK.json --results results/confirmation --out results/MATCHED_DEPTH.json
"$py" render_results.py
echo DONE
