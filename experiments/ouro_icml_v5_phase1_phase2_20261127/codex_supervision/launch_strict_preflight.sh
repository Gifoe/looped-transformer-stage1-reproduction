#!/usr/bin/env bash
set -euo pipefail
v=/root/autodl-tmp/ouro-v5-icml-phase1-20261127
source /root/autodl-tmp/ouro-v4-cfit-20261126/handoff_codex/env.sh
cd "$v"
export PYTHONPATH="$v:$v/codex_supervision:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1
exec "$py" -u codex_supervision/v5_prose_preflight.py
