#!/usr/bin/env bash
gen=${ADVANCE_GENERATION_ROOT:-/root/ouro-generative-functional-control-20261123}
source "$gen/control_env.sh"
export PYTHONPATH="$PWD:$gen:${PYTHONPATH:-}"
