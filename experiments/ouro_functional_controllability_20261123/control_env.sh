#!/usr/bin/env bash
prior_root=/root/ouro-minimal-functional-feedback-20261123
source "$prior_root/env.sh"
export PYTHONPATH="$prior_root:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=2
