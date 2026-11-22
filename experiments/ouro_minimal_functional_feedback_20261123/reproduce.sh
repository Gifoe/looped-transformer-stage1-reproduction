#!/usr/bin/env bash
set -euo pipefail
if (( $# < 4 || $# > 5 )); then
    echo 'Usage: reproduce.sh train|frozen MODEL_DIR ARC_PARQUET_DIR OUTPUT_DIR [OPENBOOKQA_RAW_PARQUET]' >&2
    exit 2
fi
mode="$1"
[[ "$mode" == train || "$mode" == frozen ]]
dir="$(cd "$(dirname "$0")" && pwd)"
experiments="$(dirname "$dir")"
arc="$experiments/ouro_arc_pareto_20261119"
prior="$experiments/ouro_semantic_amplification_arc_20261122"
original="$experiments/ouro_functional_readout_feedback_20261122"
model="$(realpath "$2")"
mkdir -p "$3" "$4"
corpus="$(realpath "$3")"
out="$(realpath "$4")"
[[ "$out" != "$dir" && "$out" != "$dir/"* ]]
py="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OPENBLAS_NUM_THREADS=4
root="$out/results"
ledger="$root/GPU_LEDGER.tsv"
legacy="$out/legacy"
external="$legacy/external/openbookqa.parquet"
mkdir -p "$root/SUBSPACE_ANALYSIS" "$legacy/results/feedback" "$legacy/external"
test ! -e "$ledger"

arc_rev=fdb98c79578d61bdf502e2ade3253c71154dd11f
for split in train validation test; do
    data="$corpus/$split.parquet"
    if [[ ! -s "$data" ]]; then
        curl --fail --location --retry 3 "https://huggingface.co/datasets/allenai/ai2_arc/resolve/$arc_rev/ARC-Challenge/ai2_arc-$split.parquet" -o "$data.tmp"
        mv "$data.tmp" "$data"
    fi
done
printf '%s  %s\n' \
    0f84cb4e393a729371c3dfd334fce14215a99dcff8095bf31271787f1e305f20 "$corpus/train.parquet" \
    a0796ca9c64d4633b595dae2b350485de9cc4f45c4adf9b59b371268f945984d "$corpus/validation.parquet" \
    b8e71916ead552c4e3bb05e3506bfbe4b5caa93b155e680fabb74dcb1ba14ec0 "$corpus/test.parquet" | sha256sum -c --quiet
raw="${5:-$legacy/external/openbookqa.raw.parquet}"
"$py" "$original/external.py" "$raw" "$external"
cp "$original/feedback.py" "$original/feedback_paths.py" "$legacy/"
cp "$original/external/openbookqa.json" "$legacy/external/"
for file in feedback/TEST_PREDICTIONS feedback/EXTERNAL_PREDICTIONS OPENBOOKQA_READOUT; do
    gzip -dc "$original/results/$file.jsonl.gz" > "$legacy/results/$file.jsonl"
done
"$py" - "$original" "$legacy" "$external" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

source, target, external = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(external) == json.loads((target / 'external/openbookqa.json').read_text())['canonical_sha256']
source, target = source / 'results', target / 'results'
lock = json.loads((source / 'feedback/FREEZE.json').read_text())
for entry, sub in ((lock, 'feedback'), (lock['readout'], '')):
    name = Path(entry['checkpoint']).name
    data = (source / sub / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry['checkpoint_sha256']
    local = target / sub / name
    local.write_bytes(data)
    entry['checkpoint'] = str(local)
(target / 'feedback/FREEZE.json').write_text(json.dumps(lock, indent=2) + '\n')
PY

remaining=18000
stage() {
    local phase="$1" start end elapsed code=0
    shift
    test "$remaining" -gt 15
    start="$(date +%s)"
    printf 'START\t%s\t%s\t%s\n' "$phase" "$start" "$remaining" >> "$ledger"
    timeout --signal=TERM --kill-after=15 "$((remaining-15))" "$py" -u "$@" > "$out/$phase.log" 2>&1 || code=$?
    end="$(date +%s)"
    elapsed=$((end-start))
    printf 'END\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$start" "$end" "$elapsed" "$code" >> "$ledger"
    remaining=$((remaining-elapsed))
    if (( code != 0 )); then
        tail -n 30 "$out/$phase.log" >&2
        exit "$code"
    fi
}

sub="$root/SUBSPACE_ANALYSIS"
sub_flags=(--archive "$arc" --model "$model" --out "$sub")
if [[ "$mode" == train ]]; then
    stage screen "$dir/train.py" screen "$model" "$arc" "$prior" "$corpus" "$root"
fi
stage gradient_arc "$dir/subspace.py" --phase fit "${sub_flags[@]}" --corpus "$corpus"
"$py" - "$sub" "$root" <<'PY'
import json
from pathlib import Path
import sys

sub, root = map(Path, sys.argv[1:])
s = json.loads((sub / 'FUNCTIONAL_GRADIENT_SPECTRUM.json').read_text())
f = s['train']['shared']['joint']['centered']['top_energy_fraction']['8']
(root / 'INIT_DECISION.json').write_text(json.dumps(dict(joint_centered_r8=f, threshold=.25, attempt=f>=.25)) + '\n')
PY
if [[ "$mode" == train ]]; then
    basis=()
    if "$py" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["attempt"])' "$root/INIT_DECISION.json"; then
        basis=(--basis "$sub/arc_basis.pt")
    fi
    stage refine "$dir/train.py" refine "$model" "$arc" "$prior" "$corpus" "$root" "${basis[@]}"
    "$py" "$dir/prepare_confirmation.py" "$root"
else
    cp "$dir/results/"*.pt "$dir/results/FREEZE.json" "$dir/results/VALIDATION.json" "$dir/results/"VAL_*.jsonl "$root/"
fi
mapfile -t arms < <("$py" - "$root/FREEZE.json" <<'PY'
import json
import sys

freeze = json.load(open(sys.argv[1]))
for name in freeze['generation_arms']:
    arm = freeze['arms'][name]
    print(name, arm['file'], arm['dose'], sep='\t')
PY
)
generation=()
for row in "${arms[@]}"; do
    IFS=$'\t' read -r name weight dose <<< "$row"
    generation+=(--minimal "$name=$root/$weight" --dose "$name=$dose")
done
IFS=$'\t' read -r name weight dose <<< "${arms[0]}"
best="$root/$weight"
stage benchmark "$dir/benchmark.py" "$model" "$arc" "$prior" "$legacy" "$corpus" "$external" "$root"
stage generation "$dir/generation.py" --model "$model" --archive "$arc" --corpus "$corpus" \
    --prior "$prior" --legacy "$legacy" --legacy-freeze "$legacy/results/feedback" \
    --external "$external" --out "$root/generation" "${generation[@]}"
"$py" "$dir/generation_compare.py" "$root"
"$py" "$dir/subspace.py" --phase align --out "$sub" --checkpoint "$best"
stage gradient_external "$dir/subspace.py" --phase external "${sub_flags[@]}" \
    --checkpoint "$best" --freeze "$root/FREEZE.json" --external-data "$external"
stage subspace_intervene "$dir/subspace.py" --phase intervene "${sub_flags[@]}" --corpus "$corpus" \
    --checkpoint "$best" --dose "$dose" --batch 8
"$py" "$dir/subspace_transfer.py" "$sub"
"$py" "$dir/subspace_transfer.py" "$sub" --external
stage causal "$dir/causal.py" both --model "$model" --archive "$arc" --corpus "$corpus" \
    --checkpoint "$best" --dose "$dose" --distill "$root/distill8_t1.pt" --constant "$root/constant_t1.pt" \
    --out "$root/CAUSAL_ANALYSIS"
"$py" "$dir/edit_geometry.py" "$root"
stage collapse_validation "$dir/collapse_diagnostic.py" --model "$model" --archive "$arc" \
    --previous "$prior" --corpus "$corpus" --out "$root"
if [[ "$mode" == train ]]; then
    stage subspace_random_init "$dir/subspace_init_control.py" --model "$model" --archive "$arc" \
        --previous "$prior" --corpus "$corpus" --out "$root"
else
    cp "$dir/results/SUBSPACE_ANALYSIS/SUBSPACE_RANDOM_INIT_CONTROLS.json" "$sub/"
fi
"$py" "$dir/report.py" "$root" "$out/REPORT_ZH.md"
