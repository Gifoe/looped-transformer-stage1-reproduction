#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-external-data-joint-training-20261124
source external_env.sh
root=/root/ouro-v3-controllability-20261126
ident=/root/ouro-loop-aware-functional-identification-20261125
transfer=/root/ouro-cross-model-functional-controllability-20261125
feedback=/root/ouro-minimal-functional-feedback-20261123
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$root/stage1_e0/code:$root/stage2_d2:$root/stage3_d1:$root/stage4_d3:$root/stage5_d6:$root/stage6_d45:$root/tools:$ident:$transfer:$feedback:$feedback/generation_followup:/root/ouro-unified-rank1-20261123:${PYTHONPATH:-}"
mkdir -p "$(dirname "/root/autodl-tmp/ouro-v5-icml-phase1-20261127/logs/wave3_gpu0_official_winners.log")"
exec "$py" -u "/root/autodl-tmp/ouro-v5-icml-phase1-20261127/p1_v5_official_winners.py" 
