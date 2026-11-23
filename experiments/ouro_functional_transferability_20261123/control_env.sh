#!/usr/bin/env bash
source /root/ouro-rank1-functional-theory-generation-20261123/control_env.sh
latest=/root/ouro-rank1-functional-theory-generation-20261123
control=/root/ouro-functional-controllability-20261123
export PYTHONPATH="$PWD:$latest:$control:$prior_root:$followup:${PYTHONPATH:-}"
