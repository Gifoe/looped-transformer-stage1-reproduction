#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source external_env.sh
test -s data/panel/PANEL.json
if [[ ${1:-} == --wait-stage1 ]]; then
    while ! grep -qx DONE run_stage1.log; do
        pid=$(cat run_stage1.pid)
        state=$(ps -p "$pid" -o stat= | tr -d ' ')
        [[ -n "$state" && "$state" != Z* ]]
        sleep 15
    done
fi
exec "$py" -u run_joint.py --python "$py" --archive "$archive" --previous "$previous" --model "$model" \
    --feedback "$feedback" --unified "$unified" --advantage "$advantage" --controllability "$controllability" \
    --generation "$gen" --historical "$historical"
