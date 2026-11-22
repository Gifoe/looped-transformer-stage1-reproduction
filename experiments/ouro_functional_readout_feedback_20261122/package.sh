#!/usr/bin/env bash
set -euo pipefail
src=/root/ouro-functional-readout-feedback-20261122
parent=/root/ouro-semantic-amplification-arc-release-20261122
repo=/root/ouro-functional-readout-feedback-release-20261122
base=652741d8c32c787a2c34b3f7abf5bbb29d38f026
base_tree=e76799bb7c83df92e792d3bfe4cba73f40912290
name=ouro_functional_readout_feedback_20261122
feature=experiments/$name
cd "$src"
source env.sh
for f in dual.log feedback_fit.log direction_audit.log full.log diagnostics.log; do test "$(tail -n 1 "$f")" = DONE; done
sha256sum -c results/DUAL_CODE_LOCK.sha256 results/DIRECTION_CODE_LOCK.sha256 \
  results/CONFIRMATION_LOCK.sha256 results/DIAGNOSTIC_CODE_LOCK.sha256
"$py" inspect_results.py results "$archive" "$previous" "$corpus" external/openbookqa.parquet
"$py" report.py results "$previous" REVIEW_ZH.md REPORT_ZH.md
if ! test -d "$repo/.git"; then
    git init -q -b research/ouro-functional-readout-feedback-20261122 "$repo"
    git -C "$repo" config gc.auto 0
    git -C "$repo" config user.name "$(git -C "$parent" config user.name)"
    git -C "$repo" config user.email "$(git -C "$parent" config user.email)"
    { printf '%s\n' "$parent/.git/objects"; cat "$parent/.git/objects/info/alternates"; } > "$repo/.git/objects/info/alternates"
    git -C "$repo" update-ref HEAD "$base"
fi
d="$repo/$feature"
mkdir -p "$d/results/feedback" "$d/external"
cp bridge.py calibration.py diagnostics.py diagnostics_initial.py direction_audit.py direction_replay.py dual.py external.py \
   feedback.py feedback_paths.py inspect_results.py readout_eval.py report.py \
   ./*.sh ./*.md semantic_audit_deck.json "$d/"
cp dual.log calibration.log feedback_fit.log direction_audit.log full.log feedback_mechanism.log feedback_test.log \
   readout_test.log feedback_external.log feedback_controls.log diagnostics.log diagnostics_initial.log "$d/"
cp external/openbookqa.json "$d/external/"
for f in results/*.json results/*.jsonl results/*.pt results/*.sha256 results/*.txt results/GPU_LEDGER.tsv \
         results/feedback/*.json results/feedback/*.jsonl results/feedback/*.pt; do
    if [[ "$f" == *.jsonl || "$f" == */OOF_UTILITY.json ]]; then
        gzip -n -c "$f" > "$d/$f.gz"
    else
        cp "$f" "$d/$f"
    fi
done
mv "$d/results/FEEDBACK_CODE_LOCK.sha256" "$d/results/FEEDBACK_FIT_CODE_HISTORY.txt"
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
commit="$(git commit-tree "$tree" -p "$base" -m 'Test functional readout and causal recurrent feedback on full ARC and OpenBookQA')"
git diff-tree --no-commit-id -r --name-status "$base" "$commit" | awk -v f="$feature/" -v n="$nfiles" 'substr($2,1,length(f))!=f || $1!="A" {bad=1} END {exit (bad || NR!=n)}'
git update-ref HEAD "$commit" "$base"
git show -s --format='%H%n%T%n%s' "$commit"
printf 'DONE\n'
