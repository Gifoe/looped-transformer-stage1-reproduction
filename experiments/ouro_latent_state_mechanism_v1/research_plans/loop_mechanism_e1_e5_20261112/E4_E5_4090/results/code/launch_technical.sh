#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ouro-e4-e5
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
OUT=repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/results/technical_full_v1
mkdir -p "$OUT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/launch_utc.txt"
printf '%s\n' 'venv/bin/python -u runner.py --inputs inputs/runner_inputs.jsonl --cohort inputs/train64_selection.json --output repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/results/technical_full_v1 --limit 2' > "$OUT/command.txt"
exec venv/bin/python -u runner.py --inputs inputs/runner_inputs.jsonl --cohort inputs/train64_selection.json --output "$OUT" --limit 2
