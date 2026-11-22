#!/usr/bin/env bash
set -euo pipefail
src="$(cd "$(dirname "$0")" && pwd)"
parent=/root/ouro-functional-readout-feedback-release-20261122
repo=/root/ouro-minimal-functional-feedback-release-20261123
branch=research/ouro-minimal-functional-feedback
base=f1b1b5fd5809ea6cba637b3de9f0acc4ce7c5085
base_tree=f9df94a0a08d97c5b81c66dcc98d905ac94be557
name=ouro_minimal_functional_feedback_20261123
feature=experiments/$name
cd "$src"
source env.sh
test "$(tail -n 1 confirm.log)" = DONE
sha256sum -c results/TRAIN_CODE_LOCK.sha256 results/SUBSPACE_FIT_LOCK.sha256 results/CONFIRMATION_LOCK.sha256
"$py" report.py results REPORT_ZH.md
if [[ ! -d "$repo/.git" ]]; then
    git init -q -b "$branch" "$repo"
    git -C "$repo" config gc.auto 0
    git -C "$repo" config user.name "$(git -C "$parent" config user.name)"
    git -C "$repo" config user.email "$(git -C "$parent" config user.email)"
    { printf '%s\n' "$parent/.git/objects"; cat "$parent/.git/objects/info/alternates"; } > "$repo/.git/objects/info/alternates"
    git -C "$repo" update-ref HEAD "$base"
fi
d="$repo/$feature"
mkdir -p "$d/results/"{SUBSPACE_ANALYSIS,CAUSAL_ANALYSIS,generation} "$d/logs"
cp benchmark.py causal.py collapse_diagnostic.py edit_geometry.py generation.py generation_compare.py minimal.py prepare_confirmation.py report.py \
   subspace.py subspace_init_control.py subspace_intervene.py subspace_transfer.py train.py ./*.sh ./*.md "$d/"
cp screen.log refine.log benchmark.log generation.log gradient_arc.log gradient_external.log subspace_intervene.log \
   causal.log confirm.log screen-first-launch.log benchmark-first-launch.log confirm-first-launch.log random_init.log collapse.log "$d/logs/"
cp results/*.json results/*.jsonl results/*.pt results/*.sha256 results/GPU_LEDGER.tsv "$d/results/"
cp results/generation/*.json "$d/results/generation/"
cp results/generation/GENERATION_RESULTS.jsonl "$d/results/GENERATION_RESULTS.jsonl"
cp results/SUBSPACE_ANALYSIS/* "$d/results/SUBSPACE_ANALYSIS/"
cp results/CAUSAL_ANALYSIS/* "$d/results/CAUSAL_ANALYSIS/"
cd "$d"
find . -type f ! -name ARTIFACTS.sha256 -print0 | sort -z | xargs -0 sha256sum > ARTIFACTS.sha256
sha256sum -c --quiet ARTIFACTS.sha256
nfiles="$(find . -type f | wc -l)"
cd "$repo"
test "$(git rev-parse HEAD)" = "$base"
test "$(git rev-parse "$base^{tree}")" = "$base_tree"
export GIT_INDEX_FILE="$repo/.git/index.feature"
git add -- "$feature"
index="$(git write-tree)"
sub="$(git rev-parse "$index:$feature")"
exp="$( { git ls-tree "$base:experiments"; printf '040000 tree %s\t%s\n' "$sub" "$name"; } | git mktree --missing)"
tree="$(git ls-tree "$base" | awk -v id="$exp" 'BEGIN {OFS="\t"} $4=="experiments" {print $1" "$2" "id,$4; next} {print}' | git mktree --missing)"
commit="$(git commit-tree "$tree" -p "$base" -m 'Evaluate minimal low-rank recurrent feedback on full ARC and OpenBookQA')"
git diff-tree --no-commit-id -r --name-status "$base" "$commit" | awk -v f="$feature/" -v n="$nfiles" 'substr($2,1,length(f))!=f || $1!="A" {bad=1} END {exit (bad || NR!=n)}'
git update-ref HEAD "$commit" "$base"
git show -s --format='%H%n%T%n%s' "$commit"
printf 'DONE\n'
