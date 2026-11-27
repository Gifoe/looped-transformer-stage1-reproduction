#!/usr/bin/env bash
set -euo pipefail
pid=$1
signature=$2
root=$3
cd "$root"
while test -r "/proc/$pid/stat" && test "$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null)" = "$signature"; do
    if test -s track_a_csbp/CSBP_PREFLIGHT_VERDICT.json; then
        if tr '\0' '\n' <"/proc/$pid/cmdline" | grep -Fxq "$root/p1_gpu1_csbp_pure_prose.py"; then
            kill -TERM "$pid"
            date --iso-8601=seconds > codex_supervision/INHERITED_TRACK_A_HELD_AT_PREFLIGHT
            sha256sum track_a_csbp/{CSBP_PREFLIGHT_VERDICT,CORPUS_DEDUP_AUDIT}.json > codex_supervision/INHERITED_TRACK_A_ASSET_SHA256
            echo "TRACK_A_INHERITED_DIAGNOSTIC_RETAINED_BEFORE_TRAINING"
        fi
        exit
    fi
    sleep 0.5
done
echo "TRACK_A_INHERITED_PID_EXITED"
