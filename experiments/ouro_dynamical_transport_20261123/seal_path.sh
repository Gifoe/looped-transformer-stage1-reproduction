#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
if [[ ! -e results/PATH_LOCK.json ]]; then
    "$py" freeze_cohorts.py --transport "$transport_archive" --out data/path --n 48 --salt transport-path-v1 \
        --extra-exclusion data/arc_confirm.jsonl --extra-exclusion data/csqa_confirm.jsonl
    "$py" - "$t1" "$transport_archive/results/HOLDOUT/qasc/source/s261123137/k234/controller.pt" <<'PY'
from pathlib import Path
import sys
from archive_audit import dump, sha
t1,qasc=map(Path,sys.argv[1:])
phases={name:dict(data_sha256=sha(Path("data/path")/f"{domain}_confirm.jsonl"),weights_sha256=sha(weight))
        for name,domain,weight in (("arc","arc",t1),("csqa","csqa",t1),("qasc","arc",qasc))}
dump(Path("results/PATH_LOCK.json"),dict(phases=phases,start=2,receive=3,end=4,cases_per_dataset=48,
    prediction="Midpoint adjoint has smaller paired working-dose absolute error than using the same native base adjoint. "
               "Compare ARC T1, CSQA T1 and QASC→ARC on a further holdout. Uses task labels in offline derivative only.",
    script_sha256=sha(Path("path_integration.py"))))
PY
    sha256sum results/PATH_LOCK.json > PATH_LOCK.sha256
fi
sha256sum --check --quiet PATH_LOCK.sha256
