#!/usr/bin/env bash
joint_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
advantage=${OURO_ADVANTAGE_ROOT:-/root/ouro-shared-functional-advantage-20261124}
source "$advantage/control_env.sh"
feedback=${OURO_FEEDBACK_ROOT:-/root/ouro-minimal-functional-feedback-20261123}
unified=${OURO_UNIFIED_ROOT:-/root/ouro-unified-rank1-20261123}
controllability=${OURO_CONTROLLABILITY_ROOT:-/root/ouro-functional-controllability-20261123}
dynamics=${OURO_DYNAMICS_ROOT:-/root/ouro-dynamical-transport-20261123}
historical=${OURO_RANK1_GENERATION_ROOT:-/root/ouro-rank1-functional-theory-generation-20261123}
export PYTHONPATH="$joint_root:$advantage:$gen:$prior_transfer:${PYTHONPATH:-}"
