#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
mode=${1:-}
if [[ "$mode" == --wait ]]; then
    for ((check=0;check<2400;check++)); do
        if grep -qx DONE run_round3.log; then break; fi
        sleep 5
    done
fi
grep -qx DONE run_round3.log
base=results/fits/mixed/controller.pt
fit=results/fits/anchored_policy
gpu_args=(--archive "$archive" --previous "$previous" --model "$model" --data data)
case "$mode" in
    --resume-evaluation)
        "$py" - "$fit" <<'PY'
from hashlib import file_digest
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
receipt = json.loads((root / "TRAIN.json").read_text())
with (root / "controller.pt").open("rb") as checkpoint:
    if file_digest(checkpoint, "sha256").hexdigest() != receipt["checkpoint_sha256"]:
        raise RuntimeError("The saved anchored controller differs from its training receipt")
if receipt["number_updates"] != 336:
    raise RuntimeError("The anchored fit did not complete its full schedule")
PY
        evaluation_log=dev_anchored_policy_resume.log
        ;;
    ''|--wait)
        bash control_budget.sh train_anchored_policy train_advantage.py "${gpu_args[@]}" --moments results/MOMENT_FIT \
            --objective worst_advantage --initial "$base" --reference "$base" --steps 336 --lr .00035 \
            --on-policy results/ON_POLICY_FIT/TRAIN_PREFIXES.jsonl --out "$fit" > train_anchored_policy.log 2>&1
        evaluation_log=dev_anchored_policy.log
        ;;
    *)
        printf 'Usage: %s [--wait|--resume-evaluation]\n' "$0" >&2
        exit 2
        ;;
esac
bash control_budget.sh dev_anchored_policy evaluate_controller.py "${gpu_args[@]}" --arm anchored_policy \
    --checkpoint "$fit/controller.pt" --depths 2 3 4 --limit 64 \
    --out results/development >> "$evaluation_log" 2>&1
echo DONE
