#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
run=$PWD
prior=/root/ouro-rank1-functional-theory-generation-release-20261123
repo=/root/ouro-functional-transferability-release-20261123
base=0666839ae1153fd02bd3293f7234dc1ad6e35268
original=23cf864ad866bbce9d0dd01ec30543f8643830ae
branch=research/ouro-functional-transferability
slug=ouro_functional_transferability_20261123
feature="experiments/$slug"
target="$repo/$feature"

test "$(git -C "$prior" rev-parse "$base^{tree}")" = "$original"
experiments=$(git -C "$prior" rev-parse "$base:experiments")
if [[ ${1:-} == --preflight ]]; then
    printf 'Verified base tree %s and experiment tree %s\n' "$original" "$experiments"
    exit 0
fi

test "$(tail -n 1 evaluate_core.log)" = DONE
test "$(tail -n 1 source_round2.log)" = DONE
test "$(tail -n 1 gamma_confirm.log)" = DONE
test "$(tail -n 1 generation_queue.log)" = DONE
test -s results/PREDICTION_AUDIT/AUDIT.json
test -s results/SPECTRUM/SPECTRUM.json
sha256sum -c --quiet PREREGISTRATION.sha256
sha256sum -c --quiet R2_ANALYSIS_LOCK.sha256
"$py" release_audit.py "$run"

mkdir -p "$repo" "$target"
g() { git -C "$repo" "$@"; }
if [[ ! -d "$repo/.git" ]]; then
    g init -b "$branch"
    g config user.name 'Ouro Research'
    g config user.email 'ouro-research@users.noreply.github.com'
    g config core.autocrlf false
fi
copy_object() {
    local kind=$1 id=$2
    test "$(git -C "$prior" cat-file "$kind" "$id" | g hash-object -w -t "$kind" --stdin)" = "$id"
}
copy_object commit "$base"
copy_object tree "$original"
copy_object tree "$experiments"
if g rev-parse --verify HEAD >/dev/null 2>&1; then
    test "$(g rev-parse HEAD)" = "$base"
else
    g update-ref HEAD "$base"
fi
test -z "$(g ls-tree "$experiments" -- "$slug")"

for source in ./*.py; do
    [[ "$source" == ./publish_api.py ]] || cp "$source" "$target/"
done
cp ./*.md ./*.sh PREREGISTRATION.sha256 R2_ANALYSIS_LOCK.sha256 GPU_LEDGER.tsv RELEASE_AUDIT.json "$target/"
mkdir -p "$target/logs"
cp ./*.log "$target/logs/"
cp -r data results "$target/"
mkdir -p "$target/reference_weights"
cp /root/ouro-unified-rank1-20261123/weights/t1.pt "$target/reference_weights/"
for loc in "$target/results/moments/"*; do
    for moment in "$loc"/*.pt; do
        gzip -n "$moment"
    done
done

cd "$target"
find . -type f ! -name ARTIFACTS.sha256 -print0 | sort -z | xargs -0 sha256sum > ARTIFACTS.sha256
sha256sum -c --quiet ARTIFACTS.sha256
test -z "$(find . -type f -size +75M -print -quit)"
count=$(find . -type f | wc -l)

cd "$repo"
export GIT_INDEX_FILE="$repo/.git/index.transfer"
g add -- "$feature"
partial=$(g write-tree)
sub=$(g rev-parse "$partial:$feature")
appended=$( { g ls-tree "$experiments"; printf '040000 tree %s\t%s\n' "$sub" "$slug"; } | g mktree --missing)
tree=$(g ls-tree "$original" | awk -v id="$appended" 'BEGIN {OFS="\t"} $4=="experiments" {print $1" "$2" "id,$4; next} {print}' | g mktree --missing)
commit=$(g commit-tree "$tree" -p "$base" -m 'Test cross-depth and cross-task transfer of frozen Ouro rank-one control')
g diff-tree --no-commit-id -r --name-status "$base" "$commit" | awk -v f="$feature/" -v n="$count" 'substr($2,1,length(f))!=f || $1!="A" {bad=1} END {exit(bad || NR!=n)}'
g update-ref HEAD "$commit" "$base"
g show -s --format='%H%n%T%n%s' "$commit"
printf 'DONE\n'
