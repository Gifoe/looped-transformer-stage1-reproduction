#!/usr/bin/env bash
set -euo pipefail
if (( $# != 4 )); then
  echo 'Usage: reproduce.sh train|frozen MODEL_DIR ARC_PARQUET_DIR OUTPUT_DIR' >&2
  exit 2
fi
mode="$1"
[[ "$mode" == train || "$mode" == frozen ]]
dir="$(cd "$(dirname "$0")" && pwd)"
parent="$(dirname "$dir")"
arc="$parent/ouro_arc_pareto_20261119"
prior="$parent/ouro_semantic_amplification_arc_20261122"
csc="$parent/ouro_causal_semantic_completion_20261122"
model="$(realpath "$2")"
corpus="$3"
out="$4"
py="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENBLAS_NUM_THREADS=4
mkdir -p "$out" "$corpus"
out="$(realpath "$out")"
[[ "$out" != "$dir" && "$out" != "$dir/"* ]]
root="$out/results"
ledger="$root/GPU_LEDGER.tsv"
mkdir -p "$root/feedback" "$out/external" "$out/prior"
test ! -e "$ledger"
cp "$prior/arc_features.py" "$prior/amplify.py" "$prior/equal_dose.py" "$out/prior/"
ln -s "$prior/results" "$out/prior/results"
arc_rev=fdb98c79578d61bdf502e2ade3253c71154dd11f
for split in train validation test; do
    if [[ ! -s "$corpus/$split.parquet" ]]; then
        curl --fail --location --retry 3 "https://huggingface.co/datasets/allenai/ai2_arc/resolve/$arc_rev/ARC-Challenge/ai2_arc-$split.parquet" -o "$corpus/$split.parquet.tmp"
        mv "$corpus/$split.parquet.tmp" "$corpus/$split.parquet"
    fi
done
test "$(sha256sum "$corpus/test.parquet" | cut -d ' ' -f1)" = b8e71916ead552c4e3bb05e3506bfbe4b5caa93b155e680fabb74dcb1ba14ec0
"$py" "$dir/external.py" "$out/external/openbookqa.raw.parquet" "$out/external/openbookqa.parquet"
test "$(sha256sum "$out/external/openbookqa.raw.parquet" | cut -d ' ' -f1)" = cd5483e366daa230c1c87bbdc512d8b7229f14f6dd04d19fc8b1a3855aaaa8a3

remaining=18000
stage() {
    phase="$1"
    shift
    test "$remaining" -gt 15
    start=$(date +%s)
    printf 'START\t%s\t%s\t%s\n' "$phase" "$start" "$remaining" >> "$ledger"
    code=0
    timeout --signal=TERM --kill-after=15 "$((remaining-15))" "$py" -u "$@" > "$out/$phase.log" 2>&1 || code=$?
    end=$(date +%s)
    dt=$((end-start))
    printf 'END\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$start" "$end" "$dt" "$code" >> "$ledger"
    remaining=$((remaining-dt))
    if (( code != 0 )); then tail -n 25 "$out/$phase.log" >&2; exit "$code"; fi
}

fb=(--model "$model" --archive "$arc" --corpus "$corpus" --prior "$out/prior" --out "$root/feedback")
if [[ "$mode" == train ]]; then
    stage features "$out/prior/arc_features.py" "$model" "$corpus" "$out/prior/features" "$arc"
    stage dual "$dir/dual.py" "$model" "$out/prior" "$arc" "$root"
    "$py" "$dir/calibration.py" "$root" "$out/prior" "$arc"
    stage feedback_fit "$dir/feedback.py" fit "${fb[@]}"
    stage distill_fit "$dir/feedback.py" distill-fit "${fb[@]}"
    mapfile -t selected < <("$py" -c 'import json,sys; c=json.load(open(sys.argv[1]))["config"]["dual"]; print(c["file"]); print(c["dose"])' "$root/READOUT_FREEZE.json")
    stage feedback_validation "$dir/feedback.py" validation "${fb[@]}" --readout "$root/${selected[0]}" --readout-dose "${selected[1]}"
    stage direction_audit "$dir/direction_audit.py" "$out/prior" "$arc" "$csc/results" "$model" "$root" --batch 12 --phase all
    "$py" "$dir/direction_replay.py" "$root"
else
    cp "$dir/results/"*.pt "$dir/results/READOUT_FREEZE.json" "$dir/results/CALIBRATION_FREEZE.json" "$root/"
    cp "$dir/results/feedback/"*.pt "$dir/results/feedback/FREEZE.json" "$root/feedback/"
    "$py" - "$root" "$prior" <<'PY'
from pathlib import Path
import hashlib,json,sys
root,prior=map(Path,sys.argv[1:])
p=root/'feedback/FREEZE.json'; v=json.loads(p.read_text())
for record,paths in ((v,(root/'feedback',prior/'results')),(v['readout'],(root,))):
    name=Path(record['checkpoint']).name
    candidates=[d/name for d in paths if (d/name).exists()]
    path=next(f for f in candidates if hashlib.sha256(f.read_bytes()).hexdigest()==record['checkpoint_sha256'])
    record['checkpoint']=str(path)
p.write_text(json.dumps(v,indent=2)+'\n')
PY
fi
stage feedback_mechanism "$dir/feedback.py" mechanism "${fb[@]}"
stage feedback_test "$dir/feedback.py" test "${fb[@]}"
stage arc_readout "$dir/readout_eval.py" "$model" "$out/prior" "$arc" "$corpus" "$root"
stage openbook_readout "$dir/readout_eval.py" "$model" "$out/prior" "$arc" "$corpus" "$root" --external "$out/external/openbookqa.parquet"
stage feedback_external "$dir/feedback.py" external "${fb[@]}" --external-data "$out/external/openbookqa.parquet"
stage feedback_controls "$dir/feedback.py" controls "${fb[@]}"
stage diagnostics "$dir/diagnostics.py" "${fb[@]}"
"$py" "$dir/inspect_results.py" "$root" "$arc" "$prior" "$corpus" "$out/external/openbookqa.parquet"
"$py" "$dir/report.py" "$root" "$prior" "$dir/REVIEW_ZH.md" "$out/REPORT_ZH.md"
