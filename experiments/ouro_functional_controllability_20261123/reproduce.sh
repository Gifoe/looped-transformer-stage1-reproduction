#!/usr/bin/env bash
set -euo pipefail
if (( $# != 4 )); then
    echo 'Usage: bash reproduce.sh MODEL_DIR ARC_CORPUS_DIR OPENBOOKQA_PARQUET OUTPUT_DIR' >&2
    exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
sibling=$(dirname "$here")
model=$(realpath "$1")
corpus=$(realpath "$2")
external=$(realpath "$3")
mkdir -p "$4"
out=$(realpath "$4")
python="${PYTHON:-python}"
archive="$sibling/ouro_arc_pareto_20261119"
previous="$sibling/ouro_semantic_amplification_arc_20261122"
published="$sibling/ouro_minimal_functional_feedback_20261123"
prior="$out/prior_assets"
result="$out/results"
export PYTHONPATH="$published:$here:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OPENBLAS_NUM_THREADS=2
mkdir -p "$prior/results/generation/fragments" "$result"

"$python" - "$published" "$prior" <<'PY'
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

study, overlay = (Path(x) for x in sys.argv[1:])
for path in (study/"results").iterdir():
    at = overlay/"results"/path.name
    if path.name != "generation" and not at.exists():
        at.symlink_to(path, target_is_directory=path.is_dir())
for path in (study/"results/generation").glob("*.json"):
    shutil.copyfile(path, overlay/"results/generation"/path.name)
raw = study/"results/GENERATION_RESULTS.jsonl"
if not raw.exists():
    raw = study/"results/generation/GENERATION_RESULTS.jsonl"
generation = overlay/"results/generation"
shutil.copyfile(raw, generation/raw.name)
parts = defaultdict(list)
for line in raw.read_text(encoding="utf-8").splitlines(keepends=True):
    x = json.loads(line)
    parts[(x["split"],x["method"],x["depth"])].append(line)
for (task, name, depth), lines in parts.items():
    path = generation/"fragments"/f"{task}__{name}__k{depth}.jsonl"
    data = "".join(lines).encode("utf-8")
    path.write_bytes(data)
    path.with_suffix(".cost.json").write_text(json.dumps(
        {"raw_sha256":hashlib.sha256(data).hexdigest()})+"\n",encoding="utf-8")
PY

elapsed=0
exec 8>"$out/gpu.lock"
flock -n 8
run() {
    local phase="$1" start end rc=0
    shift
    if [[ -f "$out/GPU_LEDGER.tsv" ]]; then
        elapsed=$(awk -F '\t' '$1=="END" {a+=$5} END {print a+0}' "$out/GPU_LEDGER.tsv")
        [[ $(tail -n1 "$out/GPU_LEDGER.tsv") != START$'\t'* ]]
    fi
    local left=$((28800-elapsed))
    test "$left" -gt 15
    start=$(date +%s)
    printf 'START\t%s\t%s\t%s\n' "$phase" "$start" "$left" >> "$out/GPU_LEDGER.tsv"
    timeout --signal=TERM --kill-after=15 "$((left-15))" "$python" -u "$@" || rc=$?
    end=$(date +%s)
    printf 'END\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$start" "$end" "$((end-start))" "$rc" >> "$out/GPU_LEDGER.tsv"
    return "$rc"
}
axis="$prior/results/CAUSAL_ANALYSIS/mean_direction_rank1.pt"
common=(--archive "$archive" --model "$model")
path_flags=(--previous "$previous" --corpus "$corpus")
A=(--previous-study "$prior" "${common[@]}" "${path_flags[@]}" --external "$external" --out "$result/A")
C=("${common[@]}" --corpus "$corpus" --direction "$axis" --out "$result/C_CAUSAL")
E=("${common[@]}" --previous "$previous" --rank8-freeze "$prior/results/FREEZE.json"
    --freeze "$result/FREEZE.json" --batch-size 16)
run B_train "$here/train_control.py" "${common[@]}" "${path_flags[@]}" --prior "$prior" --axis "$axis" --theory "$here/THEORY_HYPOTHESES.md" --out "$result"
run D_finite_dose "$here/experiment_d.py" "${common[@]}" --corpus "$corpus" --direction "$axis" --out "$result/D_FINITE_DOSE"
run confirm "$here/confirm_control.py" "${common[@]}" "${path_flags[@]}" --external "$external" --prior "$prior" --out "$result"
run A_geometry "$here/experiment_a.py" --phase geometry "${A[@]}"
run A_factorial "$here/experiment_a.py" --phase factorial "${A[@]}"
"$python" "$here/a_audit.py" "$result/A"
run C_factorial "$here/experiment_c.py" --stage factorial "${C[@]}"
run C_paths "$here/experiment_c.py" --stage paths "${C[@]}"
run A_generation "$here/experiment_a.py" --phase generation "${A[@]}"
adaptive=$("$python" -c 'import json,sys; x=json.load(open(sys.argv[1])); t=x["class_representatives"]["adaptive"]; print(x["weights"][t]["file"])' "$result/FREEZE.json")
run A_learned "$here/experiment_a.py" --phase learned --learned "$result/weights/$adaptive" "${A[@]}"
run E_semantics "$here/experiment_e.py" semantic "${E[@]}" --axis "$axis" --c-lock "$result/C_CAUSAL/LOCKED_LOCATIONS.json" --out "$result/E_SEMANTIC" --generate
run CSQA "$here/experiment_e.py" transfer "${E[@]}" --out "$result/CSQA_TRANSFER"

if [[ -n "${STAGE2_ASSETS:-}" ]] && "$python" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$result/SEVEN_TASK_GATE.json"; then
    ln -s "$(realpath "$STAGE2_ASSETS")" "$prior/stage2"
    run seven "$here/seven_control.py" "${common[@]}" --previous "$previous" --prior "$prior" --freeze "$result/FREEZE.json" --out "$result/OPTIONAL_SEVEN"
fi
