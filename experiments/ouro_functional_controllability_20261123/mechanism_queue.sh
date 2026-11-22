#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
axis="$prior_root/results/CAUSAL_ANALYSIS/mean_direction_rank1.pt"
until [[ $(tail -n 1 core_queue.log) == DONE ]]; do
    ps -p "$1" -o stat= | grep -qv Z
    sleep 12
done
do_stage() {
    local label="$1"; shift
    echo "RUN $label"
    if ! bash control_budget.sh "$label" "$@" > "$label.log" 2>&1; then
        echo "FAILED $label"
        tail -15 "$label.log"
        return 1
    fi
    echo "COMPLETE $label"
}
A=(--previous-study "$prior_root" --archive "$archive" --previous "$previous" --model "$model"
   --corpus "$corpus" --external "$external" --out results/A)
C=(--archive "$archive" --model "$model" --corpus "$corpus" --direction "$axis" --out results/C_CAUSAL)
E=(--archive "$archive" --model "$model" --previous "$previous" --rank8-freeze "$prior_root/results/FREEZE.json"
   --freeze results/FREEZE.json --batch-size 16)
do_stage geometry_a experiment_a.py --phase geometry "${A[@]}"
do_stage factorial_a experiment_a.py --phase factorial "${A[@]}"
do_stage causal_c experiment_c.py --stage factorial "${C[@]}"
do_stage paths_c experiment_c.py --stage paths "${C[@]}"
do_stage gen_a experiment_a.py --phase generation "${A[@]}"
adaptive=$("$py" -c 'import json; x=json.load(open("results/FREEZE.json")); a=x["class_representatives"]["adaptive"]; print(x["weights"][a]["file"])')
do_stage learn_a experiment_a.py --phase learned --learned "results/weights/$adaptive" "${A[@]}"
do_stage semantics_e experiment_e.py semantic "${E[@]}" --axis "$axis" --c-lock results/C_CAUSAL/LOCKED_LOCATIONS.json \
    --out results/E_SEMANTIC --generate
do_stage csqa experiment_e.py transfer "${E[@]}" --out results/CSQA_TRANSFER
printf 'DONE\n'
