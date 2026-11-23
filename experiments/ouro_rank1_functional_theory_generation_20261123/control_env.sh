#!/usr/bin/env bash
source /root/ouro-functional-controllability-20261123/control_env.sh
control=/root/ouro-functional-controllability-20261123
followup="$prior_root/generation_followup"
export PYTHONPATH="$PWD:$control:$prior_root:$followup:${PYTHONPATH:-}"
