#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source external_env.sh
work="$PWD"
repo=/root/ouro-external-data-joint-training-release-20261124
feature=experiments/ouro_external_data_joint_training_20261124
target="$repo/$feature"
grep -qx DONE run_after_stage1.log
test -s results/RESULTS.json
test -s results/FINAL_CONTROLLER.pt
test -s MECHANISM.md
g() { git -C "$repo" "$@"; }
base=$(g rev-parse HEAD)
mkdir -p "$target"/{configs,data/panel,results,logs}
for f in ./*.py; do
    if [[ "$f" != ./publish* ]]; then cp "$f" "$target/"; fi
done
cp ./*.sh ./*.md "$target/"
cp configs/study.json "$target/configs/"
cp data/{MANIFEST.json,EXCLUSIONS.jsonl,SCHEDULE_PREFLIGHT.json} "$target/data/"
cp data/panel/PANEL.json "$target/data/panel/"
cp results/{RESULTS.csv,RESULTS.json,FINAL_CONTROLLER.pt,FREEZE.json,DEVELOPMENT_SELECTION.json,CD_COMPUTE_AUDIT.json} "$target/results/"
cp GPU_LEDGER.tsv "$target/"

bundle() {
    local destination=$1; shift
    tar --mtime=@0 --owner=0 --group=0 --numeric-owner "$@" | gzip -n > "$target/$destination"
    "$py" - "$target/$destination" "$work" <<'PY'
from hashlib import file_digest
from pathlib import Path
import sys
import tarfile

target, root = map(Path, sys.argv[1:])
with tarfile.open(target,'r:gz') as archive:
    for entry in archive:
        if entry.isfile():
            with (root/entry.name).open('rb') as original, archive.extractfile(entry) as packed:
                if file_digest(original,'sha256').digest() != file_digest(packed,'sha256').digest():
                    raise RuntimeError(f'Archive contents changed: {entry.name}')
PY
}
bundle data/training_and_dev_pools.tar.gz -cf - data/*_train.jsonl data/*_development.jsonl
for group in test/core test/benchmark transfer/core dev-small/core dev-large/core; do
    dirname=${group//\//_}
    for dataset in $(find "results/$group" -maxdepth 1 -name '*.jsonl' -printf '%f\n' | awk -F '__' '{print $1}' | sort -u); do
        files=("results/$group"/"${dataset}"__*.json "results/$group"/"${dataset}"__*.jsonl)
        bundle "results/${dirname}_${dataset}.tar.gz" -cf - "${files[@]}"
    done
done
fit_dirs=(results/fits)
if [[ -d results/compute_matched_D ]]; then fit_dirs+=(results/compute_matched_D); fi
bundle results/fits.tar.gz --exclude=resume.pt -cf - "${fit_dirs[@]}" results/mechanism
bundle logs/execution.tar.gz -cf - logs run_stage1.log run_after_stage1.log prepare_data.log prepare_panel.log
( cd "$target"; find . -type f ! -name ARTIFACTS.sha256 -print0 | sort -z | xargs -0 sha256sum > ARTIFACTS.sha256; sha256sum -c --quiet ARTIFACTS.sha256 )

g add -- "$feature"
partial=$(g write-tree)
update=$(g rev-parse "$partial:$feature")
subtree=$(g ls-tree "$base:experiments" | awk -v new="$update" -v leaf="${feature##*/}" 'BEGIN{OFS="\t"} $4==leaf{print $1" "$2" "new,$4;next}{print}' | g mktree --missing)
root=$(g ls-tree "$base^{tree}" | awk -v new="$subtree" 'BEGIN{OFS="\t"} $4=="experiments"{print $1" "$2" "new,$4;next}{print}' | g mktree --missing)
commit=$(g commit-tree "$root" -p "$base" -m 'Report the complete matched Ouro joint supervision and benchmark study')
g diff-tree --no-commit-id -r --name-status "$base" "$commit" | awk -v pre="$feature/" 'substr($2,1,length(pre))!=pre || ($1!="A"&&$1!="M"){bad=1} END{exit(bad||NR<14)}'
g update-ref HEAD "$commit" "$base"
g show -s --format='%H%n%T' "$commit"
echo DONE
