#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
if [[ ${1:-} == --wait ]]; then
    for ((i=0;i<3600;i++)); do
        grep -qx DONE run_confirmation.log && break
        sleep 5
    done
fi
grep -qx DONE run_confirmation.log
export CUDA_VISIBLE_DEVICES=""
"$py" audit_release.py --verify-test --index results
"$py" paired_analysis.py --phase confirmation
"$py" paired_analysis.py --phase development_rescored
"$py" analyze_transfer_depth.py
"$py" result_tables.py
"$py" render_summary.py
echo DONE
