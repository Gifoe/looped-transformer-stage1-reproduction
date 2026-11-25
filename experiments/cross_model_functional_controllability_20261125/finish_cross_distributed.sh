#!/usr/bin/env bash
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
joint=/root/ouro-external-data-joint-training-20261124
weights=/dev/shm/ouro-cross-model-20261125
ident=/root/ouro-loop-aware-functional-identification-20261125
cd "$joint"
source external_env.sh
cd "$study"
export PYTHONPATH="$study:$ident:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
t1="$controllability/results/weights/rank1_t1.pt"
historical="$joint/t1_full/results/RESULTS.json"
for name in ouro-1.4b-thinking ouro-2.6b ouro-2.6b-thinking; do
    grep -q "^DONE $name " "logs/card_$name.log"
    test -s "results/mechanism/$name/train/fold_3.pt"
    test -s "results/mechanism/$name/finite/FINITE.json"
done
"$py" -u merge_split_loopcd.py results/loopcd/ouro-2.6b-thinking results/loopcd_aux/ouro-2.6b-thinking
"$py" - <<'PY'
from pathlib import Path
import json
models = ("huginn-0125", "ouro-1.4b-thinking", "ouro-2.6b", "ouro-2.6b-thinking")
for model in models:
    items = json.loads((Path("results/loopcd") / model / "RESULTS.json").read_text())
    assert len(items) == 8
    for task, row in items.items():
        if row["source"] == "current_matched":
            assert set(row["arms"]) == {"fixed", "adaptive"}, (model, task)
print("All expected LoopCD arms verified")
PY
"$py" -u audit_transfer_mechanism.py summarize --models MODELS.json --t1 "$t1" \
    --base-train "$ident/results/moments/arc_challenge/candidate/k4" --base-finite "$ident/results/finite/arc_k4" \
    --results results/mechanism --out results/mechanism/MECHANISM.json
"$py" -u report_transfer.py --results results --out REPORT_ZH.md --historical "$historical" --final --require-huginn
"$py" -u - "$study" "$weights" "$t1" "$historical" <<'PY'
import argparse
from pathlib import Path
import sys
import package_transfer as release
release.HUGINN_CODE += ("run_cross_card.sh", "run_cross_split_card.sh", "run_split_loopcd.py",
                       "merge_split_loopcd.py", "finish_cross_distributed.sh")
root, models, t1, historical = map(Path, sys.argv[1:])
release.pack(argparse.Namespace(root=root, models=models, t1=t1, historical=historical, only=None,
             with_huginn=True, huginn_benchmarks=False, huginn_partial=False,
             out=Path("/root/ouro-cross-model-functional-controllability-release-final-20261125")))
PY
printf 'DONE %s\n' "$(date -Is)"
