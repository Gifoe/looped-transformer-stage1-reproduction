#!/usr/bin/env bash
work=/root/autodl-tmp/ouro-cross-task-20261118
prior=/root/autodl-tmp/ouro-arc-pareto-20261119
py="$work/venv/bin/python"
model="$work/model"
corpus="$prior/data"
archive="$prior/release_git/experiments/ouro_arc_pareto_20261119"
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$work/repo/experiments/ouro_latent_state_mechanism_v1${PYTHONPATH:+:$PYTHONPATH}"
