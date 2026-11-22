#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
sha256sum arc_features.py > results/FEATURE_LOCK.sha256
bash gpu_budget.sh features arc_features.py "$model" "$corpus" features "$archive"
printf 'DONE\n'
