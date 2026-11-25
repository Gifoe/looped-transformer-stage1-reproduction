#!/usr/bin/env bash
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
joint=${OURO_EXTERNAL_ROOT:-/root/ouro-external-data-joint-training-20261124}
selection=$(realpath "${1:?pass the final development SELECTION.json}")
cd "$joint"
source external_env.sh
cd "$study"
export PYTHONPATH="$study:$PYTHONPATH"
mkdir -p logs
test -s "$selection"
release="$joint/mixed_full"
registry=(
    --archive "$archive" --previous "$previous" --model "$model"
    --selection "$selection"
    --reference-freeze "$release/results/FREEZE.json"
    --stage2 "$release/archive" --minimal-code "$feedback" --unified "$unified"
    --followup "$feedback/generation_followup" --historical "$historical"
    --panel "$joint/data/panel"
    --t1 "$controllability/results/weights/rank1_t1.pt"
    --t2 "$controllability/results/weights/rank1_t2.pt"
    --mixed "$advantage/results/fits/mixed/controller.pt"
    --rank8 "$feedback/results/linear8_t1.pt"
    --out results/official
)
"$py" -u run_full.py "${registry[@]}" --block freeze
(
    exec 8>"$gpu_lock"
    flock 8
    "$py" -u run_full.py "${registry[@]}" --block legacy > logs/official_legacy.log 2>&1
    "$py" -u run_full.py "${registry[@]}" --block extra > logs/official_extra.log 2>&1
)
"$py" -u aggregate_full.py --out results/official --release "$joint" --panel "$joint/data/panel" \
    --stage2 "$release/archive" --minimal "$feedback" --model "$model" \
    --followup "$feedback/generation_followup" --unified "$unified" --historical "$historical"
sed 's#](RESULTS\.#](results/official/RESULTS.#g' results/official/RESULTS.md > RESULTS.md.partial
mv RESULTS.md.partial RESULTS.md
"$py" -u plot_identification.py --official results/official/RESULTS.json --out figures/official
"$py" -u final_report.py --out REPORT_ZH.md
"$py" -u package_research.py --out /root/ouro-loop-aware-functional-identification-release-20261125
printf 'DONE\n'
