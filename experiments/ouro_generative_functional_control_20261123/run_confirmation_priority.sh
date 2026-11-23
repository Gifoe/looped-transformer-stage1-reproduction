#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
protocol=results/FINAL_PROTOCOL.json
sha256sum -c --quiet FINAL_PROTOCOL.sha256
arms=$("$py" -c 'import json; print(" ".join(json.load(open("results/FINAL_PROTOCOL.json"))["allowed_run_order"]))')
instruction_arms=$("$py" -c 'import json; p=json.load(open("results/FINAL_PROTOCOL.json")); order=p["allowed_run_order"]; print(" ".join(a for a in dict.fromkeys([p["selected_arm"], "native", "multitask_full", "gen_rank", "rank_only", "gen_only", *order]) if a in order))')

gpu() {
    local stage=$1 arm=$2 accounting=$3
    bash control_budget.sh "test_${accounting}_$arm" evaluate_suite.py --archive "$archive" \
        --previous "$previous" --model "$model" --protocol "$protocol" --arm "$arm" \
        --stage "$stage" --out results/confirmation >> "confirmation_${arm}.log" 2>&1
}

primary_k4() {
    local arm=$1 prefix=results/confirmation receipt job runner worker status=0
    receipt="$prefix/gsm8k__${arm}__k4__gsm_process.json"
    if [[ -s "$receipt" ]]; then return 0; fi
    # Stop only this evaluator after the last core K4 receipt. The frozen evaluator
    # writes receipts atomically and verifies them when resuming the K4 stage.
    bash control_budget.sh "test_k4_core_$arm" evaluate_suite.py --archive "$archive" \
        --previous "$previous" --model "$model" --protocol "$protocol" --arm "$arm" \
        --stage k4 --out "$prefix" >> "confirmation_${arm}.log" 2>&1 &
    job=$!
    while kill -0 "$job" 2>/dev/null && [[ ! -s "$receipt" ]]; do sleep .5; done
    if [[ -s "$receipt" ]] && runner=$(pgrep -P "$job" -x timeout) && [[ "$runner" =~ ^[0-9]+$ ]] &&
        worker=$(pgrep -P "$runner" -f 'evaluate_suite[.]py') && [[ "$worker" =~ ^[0-9]+$ ]]; then
        printf 'Core K4 receipts complete, splitting the queue for %s\n' "$arm"
        kill -TERM "$worker" 2>/dev/null || true
    fi
    wait "$job" || status=$?
    [[ -s "$receipt" && ( "$status" == 0 || "$status" == 143 ) ]]
}

# The previous foreground shell may have left an evaluator holding the shared lock.
# Wait for its ledger closure, retaining its complete results and actual GPU time.
(flock 8) 8>"$gpu_lock"
for arm in $arms; do primary_k4 "$arm"; done
for arm in $arms; do
    if [[ ! -s "results/confirmation/gsm8k__${arm}__k6__gsm_final.json" ]]; then
        gpu qa_transfer "$arm" qa_transfer
    fi
done
# Compare the full mixed fit with Native and the same-loss single-source fit first.
for arm in $instruction_arms; do
    if [[ ! -s "results/confirmation/smol-constraints__${arm}__k4__other.json" ]]; then
        gpu k4 "$arm" k4_instructions
    fi
done
for arm in $instruction_arms; do
    if [[ ! -s "results/confirmation/smol-constraints__${arm}__k6__other.json" ]]; then
        gpu instruction_transfer "$arm" instruction_transfer
    fi
done
echo DONE
