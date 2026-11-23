#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
prior=/root/ouro-functional-transferability-release-20261123
repo=/root/ouro-dynamical-transport-release-20261123
base=1b5cbdd74929e82b2b51688bfff93f37ce194015
tree=69dc864aed8da3884d9f7aad9d0b2a10118f42dc
slug=ouro_dynamical_transport_20261123
feature="experiments/$slug"
target="$repo/$feature"

test "$(git -C "$prior" rev-parse "$base^{tree}")" = "$tree"
experiment_tree=$(git -C "$prior" rev-parse "$base:experiments")
if [[ ${1:-} == --preflight ]]; then
    printf 'Verified parent tree %s\n' "$tree"
    exit 0
fi
for step in run_e2 run_e3_discovery run_confirmation run_context run_mixed_check run_negative_path run_path; do
    test "$(tail -1 "$step.log")" = DONE
done
sha256sum -c --quiet PREREGISTRATION.sha256 FORECAST_LOCK.sha256 MIXED_LOCK.sha256 PATH_LOCK.sha256
CUDA_VISIBLE_DEVICES="" "$py" audit_predictions.py .
mkdir -p "$target"
g() { git -C "$repo" "$@"; }
if [[ ! -d "$repo/.git" ]]; then
    g init -b research/ouro-dynamical-transport
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
    [[ "$file" != ./publish* ]] && cp "$file" "$target/" || true
done
cp ./*.md ./*.sh GPU_LEDGER.tsv PREREGISTRATION.sha256 FORECAST_LOCK.sha256 MIXED_LOCK.sha256 PATH_LOCK.sha256 "$target/"
cp -r data results "$target/"
mkdir -p "$target/logs"
cp ./*.log "$target/logs/"

cd "$target"
find . -type f ! -name ARTIFACTS.sha256 -print0 | sort -z | xargs -0 sha256sum > ARTIFACTS.sha256
sha256sum -c --quiet ARTIFACTS.sha256
count=$(find . -type f | wc -l)
cd "$repo"
export GIT_INDEX_FILE="$repo/.git/index.dynamics"
g add -- "$feature"
partial=$(g write-tree)
sub=$(g rev-parse "$partial:$feature")
base_experiments=$( { g ls-tree "$experiment_tree"; printf '040000 tree %s\t%s\n' "$sub" "$slug"; } | g mktree --missing)
published=$(g ls-tree "$tree" | awk -v id="$base_experiments" 'BEGIN{OFS="\t"} $4=="experiments"{print $1" "$2" "id,$4;next}{print}' | g mktree --missing)
commit=$(g commit-tree "$published" -p "$base" -m 'Measure signed transport of low-rank Ouro control across recurrent boundaries')
g diff-tree --no-commit-id -r --name-status "$base" "$commit" | awk -v pre="$feature/" -v n="$count" '$1!="A"||substr($2,1,length(pre))!=pre{bad=1}END{exit(bad||NR!=n)}'
g update-ref HEAD "$commit" "$base"
g show -s --format='%H%n%T%n%s' "$commit"
printf 'DONE\n'
