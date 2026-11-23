#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh

if grep -qx DONE run_confirmation.log; then
    echo 'Full coverage should use run_analysis.sh.' >&2
    exit 1
fi
(flock -n 8) 8>"$gpu_lock"
used=$(awk -F '\t' '$1=="END"{s+=$5} END{print s+0}' GPU_LEDGER.tsv)
if (( used < 17985 )); then
    echo 'The five-hour budget still has time remaining.' >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=""
"$py" audit_release.py --verify-test --allow-budget-shortfall --index results
"$py" paired_analysis.py --phase confirmation
"$py" paired_analysis.py --phase development_rescored
"$py" analyze_transfer_depth.py
"$py" result_tables.py
"$py" render_summary.py
echo DONE_BUDGET_LIMITED
