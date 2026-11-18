#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
study=$(realpath -m "${1:?absolute study output directory}")
model=$(realpath "${2:?verified Ouro model directory}")
data=$(realpath "${3:?source data directory}")
prior=$(realpath "${4:?prior semantic_interventions.jsonl}")
py=${OURO_PYTHON:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$study/logs"
one() {
  local logfile=$1
  shift
  "$py" -u "$@" 2>&1 | tee "$logfile"
}
collect() {
  local job=$1 dest=$2
  one "$dest/logs/$job.log" "$here/collect.py" "$job" --study "$dest" --model "$model"
}
intervene() {
  local job=$1 task=$2 dest=$3
  one "$dest/logs/$job-$task.log" "$here/intervene.py" "$job" "$task" --study "$dest" --model "$model"
}
one "$study/logs/prepare.log" "$here/collect.py" prepare --study "$study" --model "$model" --data "$data" --prior "$prior"
collect calibration "$study" & pid_a=$!
collect benchmarks "$study" & pid_b=$!
wait "$pid_a"
wait "$pid_b"
"$py" "$here/freeze_protocol.py" "$study"
(
  intervene A relation "$study"
  intervene main logic "$study"
  intervene C logic "$study"
) & pid_a=$!
(
  intervene B relation "$study"
  intervene main relation "$study"
  intervene main arithmetic "$study"
  intervene C arithmetic "$study"
) & pid_b=$!
(
  collect controlled "$study"
  d="$study/competence"
  mkdir -p "$d/logs"
  one "$d/logs/prepare.log" "$here/competence.py" prepare --study "$d" --base "$study" --model "$model" --data "$data"
  collect calibration "$d"
  one "$d/logs/freeze.log" "$here/competence.py" freeze --study "$d" --base "$study"
  intervene main arithmetic "$d"
  intervene C arithmetic "$d"
  intervene main logic "$d"
  intervene C logic "$d"
) & pid_c=$!
for pid in "$pid_a" "$pid_b" "$pid_c"; do wait "$pid"; done
d="$study/boolean"
mkdir -p "$d/logs"
one "$d/logs/prepare.log" "$here/booleans.py" prepare --study "$d" --model "$model"
collect calibration "$d"
one "$d/logs/freeze.log" "$here/booleans.py" freeze --study "$d"
one "$study/logs/renaming.log" "$here/renaming.py" "$study" --model "$model" & pid_a=$!
(
  one "$study/logs/screen_prepare.log" "$here/screening.py" prepare --study "$study" --model "$model"
  one "$study/logs/screen_natural.log" "$here/screening.py" screen --study "$study" --model "$model"
  for task in logic arithmetic; do
    if grep -q "\"task\": \"$task\"" "$study/screened/inputs.jsonl"; then
      mkdir -p "$study/screened/logs"
      intervene C "$task" "$study/screened"
    fi
  done
) & pid_b=$!
wait "$pid_a"
wait "$pid_b"
"$py" "$here/summarize.py" "$study"
"$py" "$here/audit_expanded.py" "$study"
"$py" "$here/plots.py" "$study"
