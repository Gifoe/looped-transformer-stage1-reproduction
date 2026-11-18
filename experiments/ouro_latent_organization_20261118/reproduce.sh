#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
study=$(realpath -m "${1:?absolute output path}")
model=$(realpath "${2:?verified model snapshot}")
sources=$(realpath "${3:?source data folder}")
prior=$(realpath "${4:?expanded study results}")
py=${OURO_PYTHON:-python}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
mkdir -p "$study/logs"
one() {
 local logfile=$1
 shift
 "$py" -u "$@" 2>&1 | tee "$study/logs/$logfile.log"
}
one prepare "$here/manifest_org.py" "$study" --model "$model" --sources "$sources" \
  --prior "$prior/inputs.jsonl" "$prior/competence/inputs.jsonl" "$prior/screened/pool.jsonl"
for task in relation logic arithmetic prontoqa arc hellaswag; do
 one "pilot-$task" "$here/capture_org.py" "$study" --model "$model" --task "$task" --pilot
done
capture() {
 one "natural-$1" "$here/capture_org.py" "$study" --model "$model" --task "$1"
}
( capture relation; capture arc ) & p=$!
( capture logic; capture prontoqa ) & q=$!
( capture arithmetic; capture hellaswag ) & r=$!
wait "$p"; wait "$q"; wait "$r"
one analysis-freeze "$here/analyze_org.py" freeze "$study"
one analysis-confirm "$here/analyze_org.py" confirm "$study" & a=$!
one causal-relation "$here/causal_org.py" "$study" --model "$model" --task relation & b=$!
one causal-logic "$here/causal_org.py" "$study" --model "$model" --task logic & c=$!
one causal-arithmetic "$here/causal_org.py" "$study" --model "$model" --task arithmetic & d=$!
one full-token "$here/full_org.py" "$study" --model "$model"
one extras "$here/extras_org.py" "$study" & e=$!
wait "$a"; wait "$b"; wait "$c"; wait "$d"; wait "$e"
one width-control "$here/width_org.py" "$study" --model "$model" & a=$!
one route-retest "$here/route_org.py" "$study" --model "$model" & b=$!
wait "$a"; wait "$b"
one aggregation "$here/post_org.py" "$study"
one audit "$here/audit_org.py" "$study" --include-server-tensors
one figures "$here/plots_org.py" "$study"
