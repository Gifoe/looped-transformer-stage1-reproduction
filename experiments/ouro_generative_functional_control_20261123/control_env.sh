#!/usr/bin/env bash
prior_transfer=${GEN_TRANSFER_ROOT:-/root/ouro-functional-transferability-20261123}
if [[ -n ${GEN_EXPERIMENTS_ROOT:-} ]]; then
    : "${GEN_MODEL:?Set GEN_MODEL to the verified Ouro snapshot}"
    : "${GEN_GPU_LOCK:?Set GEN_GPU_LOCK to a shared, writable GPU lock path}"
    archive="$GEN_EXPERIMENTS_ROOT/ouro_arc_pareto_20261119"
    previous="$GEN_EXPERIMENTS_ROOT/ouro_semantic_amplification_arc_20261122"
    prior_root="$GEN_EXPERIMENTS_ROOT/ouro_minimal_functional_feedback_20261123"
    latest="$GEN_EXPERIMENTS_ROOT/ouro_rank1_functional_theory_generation_20261123"
    control="$GEN_EXPERIMENTS_ROOT/ouro_functional_controllability_20261123"
    model=$GEN_MODEL
    py=${GEN_PYTHON:-python}
    export PYTHONPATH="$prior_root:$latest:$control:$previous:$GEN_EXPERIMENTS_ROOT/ouro_latent_state_mechanism_v1:${PYTHONPATH:-}"
else
    source "$prior_transfer/control_env.sh"
fi
gpu_lock=${GEN_GPU_LOCK:-$prior_root/results/gpu.lock}
export PYTHONPATH="$PWD:$prior_transfer:${PYTHONPATH:-}"
