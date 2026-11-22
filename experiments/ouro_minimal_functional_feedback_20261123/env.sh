#!/usr/bin/env bash
work=/root/autodl-tmp/ouro-cross-task-20261118
data=/root/autodl-tmp/ouro-arc-pareto-20261119
previous=/root/ouro-semantic-amplification-arc-20261122
legacy=/root/ouro-functional-readout-feedback-20261122
py="$work/venv/bin/python"
model="$work/model"
corpus="$data/data"
archive="$data/release_git/experiments/ouro_arc_pareto_20261119"
external="$legacy/external/openbookqa.parquet"
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$work/repo/experiments/ouro_latent_state_mechanism_v1${PYTHONPATH:+:$PYTHONPATH}"
