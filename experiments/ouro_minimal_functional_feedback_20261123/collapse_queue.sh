#!/usr/bin/env bash
set -euo pipefail
source env.sh
bash gpu_budget.sh collapse_validation collapse_diagnostic.py --model "$model" --archive "$archive" --previous "$previous" --corpus "$corpus" --out results
