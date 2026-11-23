#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
prior=/root/ouro-dynamical-transport-release-20261123
repo=${GEN_RELEASE_REPO:-/root/ouro-generative-functional-control-release-20261123}
run=$PWD
base=f2d2d46e519c3997c60cfbcfd126b59adac35641
tree=dc2c0226d6f3ea12322f3f294ab61a80c4764f17
slug=ouro_generative_functional_control_20261123
feature="experiments/$slug"
target="$repo/$feature"

test "$(git -C "$prior" rev-parse "$base^{tree}")" = "$tree"
experiment_tree=$(git -C "$prior" rev-parse "$base:experiments")
if [[ ${1:-} == --preflight ]]; then
    printf 'Verified base experiment tree %s\n' "$experiment_tree"
    exit 0
fi
for step in run_core_b run_development run_scale_up run_round4_and_freeze; do
    grep -qx DONE "$step.log"
done
sha256sum -c --quiet PREREGISTRATION.sha256 FINAL_PROTOCOL.sha256
coverage=()
if [[ ${1:-} == --snapshot ]]; then
    test -s SNAPSHOT.md
    coverage=(--allow-budget-shortfall)
elif [[ ${1:-} == --budget-limited ]]; then
    grep -qx DONE_BUDGET_LIMITED finalize_budget_limited.log
    coverage=(--allow-budget-shortfall)
else
    grep -qx DONE run_confirmation.log
fi
CUDA_VISIBLE_DEVICES="" "$py" audit_release.py --verify-test --index "$run/results" "${coverage[@]}"
test -s results/CONFIRMATION.csv
for doc in DATASET METHOD TRAINING GENERATION_RESULTS ABLATIONS REPORT_ZH REPRODUCE; do
    test -s "$doc.md"
done

mkdir -p "$target"/{data/filtered,results,logs}
g() { git -C "$repo" "$@"; }
if [[ ! -d "$repo/.git" ]]; then
    g init -b research/ouro-generative-functional-control
    g config user.name 'Ouro Research'
    g config user.email 'ouro-research@users.noreply.github.com'
    g config core.autocrlf false
fi
copy() {
    local kind=$1 id=$2
    test "$(git -C "$prior" cat-file "$kind" "$id" | g hash-object -w -t "$kind" --stdin)" = "$id"
}
copy commit "$base"
copy tree "$tree"
copy tree "$experiment_tree"
if g rev-parse --verify HEAD >/dev/null 2>&1; then
    test "$(g rev-parse HEAD)" = "$base"
else
    g update-ref HEAD "$base"
fi
test -z "$(g ls-tree "$experiment_tree" -- "$slug")"
for file in ./*.py; do
    if [[ "$file" != ./publish* ]]; then cp "$file" "$target/"; fi
done
cp ./*.md ./*.sh GPU_LEDGER.tsv PREREGISTRATION.sha256 FINAL_PROTOCOL.sha256 "$target/"
cp data/{DATA_LOCK,SCALE_LOCK}.json "$target/data/"
cp data/filtered/CONTAMINATION.json "$target/data/filtered/"
cp results/*.json results/CONFIRMATION.csv results/FINAL_CONTROLLER.pt "$target/results/"
cp ./*.log "$target/logs/"
cp -r results/PREFLIGHT "$target/results/"
for arm in results/*/TRAIN.json; do
    folder=$(dirname "$arm")
    mkdir -p "$target/$folder"
    cp "$folder"/controller.pt "$folder"/*.json "$target/$folder/"
done
historical="$prior_transfer/results/HOLDOUT/csqa/source/s261123137/k234/controller.pt"
mkdir -p "$target/results/rank_only"
cp "$historical" "$target/results/rank_only/"
for phase in development development_rescored confirmation; do
    # Deterministic archive preserves original prediction bytes and score receipts.
    find "results/$phase" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' \) -print0 |
        while IFS= read -r -d '' file; do
            if [[ "$file" != *.jsonl || -s "${file%.jsonl}.json" ]]; then printf '%s\0' "$file"; fi
        done | sort -z | tar --mtime=@0 --owner=0 --group=0 --numeric-owner --null -T - -cf - |
        gzip -n > "$target/results/$phase.tar.gz"
    "$py" - "$target/results/$phase.tar.gz" <<'PY'
from hashlib import file_digest
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:gz") as archive:
    for member in archive:
        with open(member.name, "rb") as current, archive.extractfile(member) as packaged:
            if file_digest(current, "sha256").digest() != file_digest(packaged, "sha256").digest():
                raise RuntimeError(f"Prediction packaging mismatch {member.name}")
PY
done

cd "$target"
find . -type f ! -name ARTIFACTS.sha256 -print0 | sort -z | xargs -0 sha256sum > ARTIFACTS.sha256
sha256sum -c --quiet ARTIFACTS.sha256
count=$(find . -type f | wc -l)
export GIT_INDEX_FILE="$repo/.git/index.generative"
g add -- "$feature"
partial=$(g write-tree)
sub=$(g rev-parse "$partial:$feature")
base_experiments=$( { g ls-tree "$experiment_tree"; printf '040000 tree %s\t%s\n' "$sub" "$slug"; } | g mktree --missing)
published=$(g ls-tree "$tree" | awk -v id="$base_experiments" 'BEGIN{OFS="\t"} $4=="experiments"{print $1" "$2" "id,$4;next}{print}' | g mktree --missing)
commit=$(g commit-tree "$published" -p "$base" -m 'Retrain 4096-parameter Ouro control for free generation')
g diff-tree --no-commit-id -r --name-status "$base" "$commit" | awk -v pre="$feature/" -v n="$count" '$1!="A"||substr($2,1,length(pre))!=pre{bad=1}END{exit(bad||NR!=n)}'
g update-ref HEAD "$commit" "$base"
g show -s --format='%H%n%T%n%s' "$commit"
printf 'DONE\n'
