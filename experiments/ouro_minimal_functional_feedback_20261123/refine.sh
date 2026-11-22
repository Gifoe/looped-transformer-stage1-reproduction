#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
until [[ $(tail -n 1 screen.log) == DONE ]]; do
    ps -p "${1:?screen PID}" -o stat= | grep -qv Z
    sleep 7
done
sha256sum -c results/TRAIN_CODE_LOCK.sha256
sha256sum subspace.py > results/SUBSPACE_FIT_LOCK.sha256
if [[ ! -s results/SUBSPACE_ANALYSIS/FUNCTIONAL_GRADIENT_SPECTRUM.json ]]; then
    bash gpu_budget.sh gradient_arc subspace.py --phase fit --archive "$archive" --corpus "$corpus" --model "$model" --out results/SUBSPACE_ANALYSIS > gradient_arc.log 2>&1
fi
bas=()
spectrum=results/SUBSPACE_ANALYSIS/FUNCTIONAL_GRADIENT_SPECTRUM.json
"$py" -c 'import json,sys; from pathlib import Path; f=json.load(open(sys.argv[1])); value=f["train"]["shared"]["joint"]["centered"]["top_energy_fraction"]["8"]; Path("results/INIT_DECISION.json").write_text(json.dumps(dict(joint_centered_r8=value,threshold=.25,attempt=value>=.25), indent=2)+"\n"); print(value)' "$spectrum"
if "$py" -c 'import json,sys; sys.exit(not json.load(open("results/INIT_DECISION.json"))["attempt"])'; then
    bas=(--basis results/SUBSPACE_ANALYSIS/arc_basis.pt)
fi
bash gpu_budget.sh refine train.py refine "$model" "$archive" "$previous" "$corpus" results "${bas[@]}"
printf 'DONE\n'
