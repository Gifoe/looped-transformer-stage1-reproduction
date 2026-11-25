#!/usr/bin/env bash
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
joint=${OURO_EXTERNAL_ROOT:-/root/ouro-external-data-joint-training-20261124}
cd "$joint"
source external_env.sh
cd "$study"
export PYTHONPATH="$study:$PYTHONPATH"
mkdir -p logs

while ! grep -qx DONE run_within_source.log; do
    worker=$(cat run_within_source.pid)
    if [[ ! -e "/proc/$worker/cwd" ]]; then
        echo 'The prerequisite within-source controls did not finish' >&2
        exit 3
    fi
    sleep 15
done

forecast=results/forecast_all_sources
ranking=results/FINAL_SOURCE_RANKING.json
"$py" - "$forecast" "$ranking" <<'PY'
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import minimal

root, path = map(Path, sys.argv[1:])
plan = json.loads((root / "FORECAST.json").read_text())
audit = json.loads((root / "AUDIT.json").read_text())
if audit["forecast_sha256"] != minimal.sha(root / "FORECAST.json"):
    raise ValueError("The independent ten-source forecast was not verified")
measures = defaultdict(list)
for row in plan["records"]:
    if row["scope"] == "leave_source_out" and plan["sources"][row["source"]]["mode"] == "candidate":
        measures[row["source"]].append(row["predicted"]["loop_factor_influence_warm_t1"])
values = {source: statistics.mean(scores) for source, scores in measures.items()}
ranked = sorted(values, key=lambda s: (-values[s], s))
selection = dict(forecast_sha256=audit["forecast_sha256"], chosen=ranked[:3],
                 scores=values, rule="top three candidate sources by prospective T1 tangent influence")
if path.exists() and json.loads(path.read_text()) != selection:
    raise ValueError("The final pretraining source ranking was already sealed")
minimal.dump(path, selection)
print("SEALED", " ".join(selection["chosen"]), flush=True)
PY
readarray -t shortlist < <("$py" -c 'import json,sys; print(*json.load(open(sys.argv[1]))["chosen"],sep="\n")' "$ranking")
runtime=(--archive "$archive" --previous "$previous" --model "$model")
t1="$controllability/results/weights/rank1_t1.pt"
t2="$controllability/results/weights/rank1_t2.pt"
mixed="$advantage/results/fits/mixed/controller.pt"
arms=(--weight "t1=$t1" --weight "t2=$t2" --weight "mixed=$mixed")
(
    exec 8>"$gpu_lock"
    flock 8
    for method in uniform tangent loss_tangent; do
        label="coverage_$method"
        dst="results/fits/$label"
        if [[ ! -s "$dst/TRAIN.json" ]]; then
            echo START "$label" "$(date -Is)"
            "$py" -u train_full_coverage.py "${runtime[@]}" --data data/pools_complete \
                --moments results/filtered_moments --forecast "$forecast" --include "${shortlist[@]}" \
                --initialize "$t1" --arm "$method" --out "$dst" > "logs/fit_$label.log" 2>&1
        fi
        arms+=(--weight "$label=$dst/controller.pt")
    done
    if [[ ! -s results/dev_coverage/SELECTION.json ]]; then
        "$py" -u evaluate_development.py "${runtime[@]}" --panel "$joint/data/panel" \
            --stage2 "$joint/mixed_full/archive" --mmlu "$feedback/stage2/zero_shot/mmlu_cache" \
            --unified "$unified" --out results/dev_coverage "${arms[@]}" > logs/development_coverage.log 2>&1
    fi
)
"$py" -u combine_development.py --development results/dev_refinement results/dev_within_source results/dev_coverage \
    --fits results/fits --out results/final_development
bash run_final_confirm.sh results/final_development/SELECTION.json
printf 'DONE\n'
