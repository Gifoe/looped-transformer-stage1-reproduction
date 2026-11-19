#!/usr/bin/env bash
set -euo pipefail

# Run from a new output directory. Provide the original pinned checkpoint and the
# official HotpotQA distractor validation parquet named in the data provenance.
if [[ $# != 4 ]]; then
    printf 'Usage: %s PYTHON CHECKPOINT HOTPOT_PARQUET NEW_OUTPUT_DIR\n' "$0" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
python="$1"
checkpoint="$2"
hotpot="$3"
out="$4"
export OURO_RAW_ATLAS="${OURO_RAW_ATLAS:-$out/raw_bf16.bin}"
mkdir -- "$out"

"$python" "$script_dir/mechanism_run.py" prepare "$checkpoint" "$out/tableaux.jsonl" "$out/pilot"
"$python" "$script_dir/mechanism_run.py" pilot "$checkpoint" "$out/tableaux.jsonl" "$out/pilot"
"$python" "$script_dir/mechanism_run.py" tableaux "$checkpoint" "$out/tableaux.jsonl" "$out/atlas"
"$python" "$script_dir/mechanism_hotpot.py" "$checkpoint" "$hotpot" "$out/hotpot"
"$python" "$script_dir/mechanism_hotpot_hard.py" "$checkpoint" "$hotpot" "$out/hotpot/hotpot_inputs.jsonl" "$out/hard"
"$python" "$script_dir/mechanism_fork_followup.py" "$checkpoint" "$out/tableaux.jsonl" "$out/hotpot/hotpot_inputs.jsonl" "$out/forks"
"$python" "$script_dir/mechanism_analyze.py" "$out/tableaux.jsonl" "$out/atlas" "$out/hotpot" "$out/results" "$out/figures" \
    --hard "$out/hard" --fork-followup "$out/forks"
