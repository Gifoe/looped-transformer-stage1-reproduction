#!/usr/bin/env bash
transport_archive=${TRANSPORT_ARCHIVE:-/root/ouro-functional-transferability-20261123}
source "$transport_archive/control_env.sh"
export PYTHONPATH="$PWD:$transport_archive:${PYTHONPATH:-}"
t1=${TRANSPORT_T1:-/root/ouro-unified-rank1-20261123/weights/t1.pt}
