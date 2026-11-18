#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
phase=${1:?Usage: reproduce.sh PHASE STUDY MODEL SOURCE_DATA}
study=$(realpath -m "${2:?study output directory required}")
model=$(realpath "${3:?verified model snapshot directory required}")
data=$(realpath "${4:?source data directory required}")
py=${OURO_PYTHON:-python}
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
mkdir -p "$study/logs"
inputs="$study/semantic_interventions.jsonl"
freeze="$study/site_freeze.json"
case "$phase" in
  prepare)
    sha256sum "$data/arc.parquet" "$data/hellaswag.parquet" "$data/prontoqa.zip" |
      awk '{sub(/^.*\//, "", $2); print $1 "  " $2}' > "$study/logs/source_hashes.txt"
    "$py" "$here/run.py" prepare --data "$data" --inputs "$inputs" --model "$model" --out "$study" 2>&1 |
      tee "$study/logs/prepare.log"
    ;;
  pilot|dev|confirm|trajectory)
    dest="$study/$phase"
    [[ "$phase" == confirm ]] && dest="$study/confirmation"
    args=("$phase" --inputs "$inputs" --out "$dest" --model "$model")
    [[ "$phase" == confirm || "$phase" == trajectory ]] && args+=(--freeze "$freeze")
    TIMEFORMAT='wall_seconds=%R user_seconds=%U system_seconds=%S'
    { time "$py" -u "$here/run.py" "${args[@]}"; } 2>&1 | tee "$study/logs/$phase.log"
    if [[ "$phase" == trajectory ]]; then
      mv "$dest/trajectory_results" "$study/trajectory_results"
    fi
    ;;
  freeze)
    "$py" "$here/analyze.py" freeze "$study" 2>&1 | tee "$study/logs/freeze.log"
    ;;
  results)
    "$py" "$here/analyze.py" results "$study" 2>&1 | tee "$study/logs/results.log"
    "$py" "$here/figures.py" "$study" 2>&1 | tee "$study/logs/figures.log"
    ;;
  *)
    printf 'Unknown phase: %s\n' "$phase" >&2
    exit 2
    ;;
esac
