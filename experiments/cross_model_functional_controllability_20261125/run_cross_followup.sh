#!/usr/bin/env bash
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
joint=${OURO_EXTERNAL_ROOT:-/root/ouro-external-data-joint-training-20261124}
weight_root=${OURO_TRANSFER_MODELS:-/dev/shm/ouro-cross-model-20261125}
cd "$joint"
source external_env.sh
cd "$study"
export PYTHONPATH="$study:$PYTHONPATH"

wait_for() {
    local output=$1 pidfile=$2
    while [[ ! -s "$output" ]]; do
        if ! [[ -s "$pidfile" ]] || ! ps -p "$(cat "$pidfile")" -o stat= | grep -q '^[DRSTI]'; then
            printf 'Upstream worker stopped before %s\n' "$output" >&2
            return 1
        fi
        sleep 20
    done
}

first=/root/ouro-cross-model-functional-controllability-release-20261125/RELEASE_MANIFEST.json
wait_for "$first" run_cross_models.pid
"$py" - "$first" <<'PY'
import json
import sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text())
if p["complete_ouro_models"] != 3 or p["official_benchmarks"] != 24:
    raise ValueError("The three Ouro comparisons must complete first")
PY

wait_for "$weight_root/huginn-0125/MODEL_LOCK.json" fetch_huginn.pid
printf 'START HUGINN %s\n' "$(date -Is)"
bash run_huginn.sh
printf 'START LOOPCD %s\n' "$(date -Is)"
bash run_loopcd.sh
printf 'START OURO MECHANISM %s\n' "$(date -Is)"
bash run_transfer_mechanism.sh
historical="$joint/t1_full/results/RESULTS.json"
"$py" -u report_transfer.py --results results --out REPORT_ZH.md --historical "$historical" --final --require-huginn
"$py" -u package_transfer.py --root "$study" --models "$weight_root" --t1 "$controllability/results/weights/rank1_t1.pt" \
    --historical "$historical" --out /root/ouro-cross-model-functional-controllability-release-final-20261125 --with-huginn
printf 'DONE %s\n' "$(date -Is)"
