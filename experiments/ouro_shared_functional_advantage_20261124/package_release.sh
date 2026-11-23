#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
repo=/root/ouro-shared-functional-advantage-release-20261124
prior=/root/ouro-generative-functional-control-final-release-20261123
base=36ef8cdfc86b55c969e13e262515042a8776bfde
tree=b93c5b874b1cf27f730089c48e09e99fa393c396
slug=ouro_shared_functional_advantage_20261124
feature="experiments/$slug"
target="$repo/$feature"
branch=research/ouro-shared-functional-advantage
message='Measure shared rank-one functional advantage in Ouro'
checksums=(PROTOCOL.sha256)
phases=(development FINITE_REPLAY EARLY_REPLAY)

test "$(git -C "$prior" rev-parse "$base^{tree}")" = "$tree"
experiment_tree=$(git -C "$prior" rev-parse "$base:experiments")
test -z "$(git -C "$prior" ls-tree "$experiment_tree" -- "$slug")"
if [[ ${1:-} == --preflight ]]; then
    printf 'Verified parent %s\n' "$tree"
    exit 0
fi
stages=(run_mechanism run_fitting run_development run_round3)
docs=(THEORY MECHANISM METHOD REPORT_ZH REPRODUCE)
if [[ ${1:-} == --development-snapshot ]]; then
    repo+=-snapshot
    target="$repo/$feature"
    branch+=-snapshot
    message='Archive completed Ouro functional-advantage development results'
    docs+=(DEVELOPMENT_RESULTS DEVELOPMENT_ABLATIONS)
    test -s results/SNAPSHOT_STATUS.json
    test ! -e results/CONFIRMATION_LOCK.json
else
    stages+=(run_confirmation)
    docs+=(ABLATIONS RESULTS)
    checksums+=(CONFIRMATION.sha256)
    phases+=(confirmation)
    test -s results/PAIRED_CONFIRMATION.json
    test -s results/FINAL_CONTROLLER.pt
fi
for stage in "${stages[@]}"; do
    grep -qx DONE "$stage.log"
done
sha256sum -c --quiet "${checksums[@]}"
for doc in "${docs[@]}"; do test -s "$doc.md"; done
test -s results/PAIRED_DEVELOPMENT.json

mkdir -p "$target"/{data,results,logs}
g() { git -C "$repo" "$@"; }
if [[ ! -d "$repo/.git" ]]; then
    g init -b "$branch"
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

for file in ./*.py; do
    if [[ "$file" != ./publish* ]]; then cp "$file" "$target/"; fi
done
cp ./*.md ./*.sh GPU_LEDGER.tsv "${checksums[@]}" "$target/"
cp ./*.log "$target/logs/"
cp data/DATA_LOCK.json data/confirmation/CONTAMINATION.json "$target/data/"
cp results/*.json results/*.csv results/*.pt "$target/results/"
cp -r results/{FIT_SHARED,FIT_CANDIDATE,PREFIX_CAUSALITY,ON_POLICY_FIT} "$target/results/"
for folder in results/MOMENT_{FIT,HOLDOUT} results/SAMPLE_MOMENTS_* results/fits/*; do
    mkdir -p "$target/$folder"
    for artifact in "$folder"/*.json "$folder"/controller.pt; do
        if [[ -f "$artifact" ]]; then cp "$artifact" "$target/$folder/"; fi
    done
done
for phase in "${phases[@]}"; do
    find "results/$phase" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' \) -print0 |
        sort -z | tar --mtime=@0 --owner=0 --group=0 --numeric-owner --null -T - -cf - |
        gzip -n > "$target/results/$phase.tar.gz"
    "$py" - "$target/results/$phase.tar.gz" <<'PY'
from hashlib import file_digest
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:gz") as archive:
    for member in archive:
        with open(member.name, "rb") as source, archive.extractfile(member) as saved:
            if file_digest(source, "sha256").digest() != file_digest(saved, "sha256").digest():
                raise RuntimeError(f"Packaging changed {member.name}")
PY
done

cd "$target"
find . -type f ! -name ARTIFACTS.sha256 -print0 | sort -z | xargs -0 sha256sum > ARTIFACTS.sha256
sha256sum -c --quiet ARTIFACTS.sha256
count=$(find . -type f | wc -l)
export GIT_INDEX_FILE="$repo/.git/index.advantage"
g add -- "$feature"
partial=$(g write-tree)
sub=$(g rev-parse "$partial:$feature")
parents=$( { g ls-tree "$experiment_tree"; printf '040000 tree %s\t%s\n' "$sub" "$slug"; } | g mktree --missing)
published=$(g ls-tree "$tree" | awk -v id="$parents" 'BEGIN{OFS="\t"} $4=="experiments"{print $1" "$2" "id,$4;next}{print}' | g mktree --missing)
commit=$(g commit-tree "$published" -p "$base" -m "$message")
g diff-tree --no-commit-id -r --name-status "$base" "$commit" | awk -v pre="$feature/" -v n="$count" '$1!="A"||substr($2,1,length(pre))!=pre{bad=1}END{exit(bad||NR!=n)}'
g update-ref HEAD "$commit" "$base"
g show -s --format='%H%n%T%n%s' "$commit"
echo DONE
