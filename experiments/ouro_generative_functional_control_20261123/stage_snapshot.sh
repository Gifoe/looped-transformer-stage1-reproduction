#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source control_env.sh
root=$PWD
sha256sum -c --quiet FINAL_PROTOCOL.sha256 PREREGISTRATION.sha256
stamp=$(date -Iseconds)
snap=$(mktemp -d -p /root ouro-generative-functional-control-snapshot-XXXXXX)
mkdir -p "$snap/results/confirmation"
cp ./*.py ./*.sh ./*.md ./*.log GPU_LEDGER.tsv FINAL_PROTOCOL.sha256 PREREGISTRATION.sha256 "$snap/"
ln -s "$root/data" "$snap/data"
cp results/*.json results/FINAL_CONTROLLER.pt "$snap/results/"
cp -r results/{PREFLIGHT,development,development_rescored} "$snap/results/"
for file in results/*/TRAIN.json; do
    folder=$(dirname "$file")
    mkdir -p "$snap/$folder"
    cp "$folder"/*.json "$folder/controller.pt" "$snap/$folder/"
done
for receipt in results/confirmation/*.json; do
    cp "$receipt" "${receipt%.json}.jsonl" "$snap/results/confirmation/"
done

cd "$snap"
export PYTHONPATH="$snap:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=""
"$py" audit_release.py --verify-test --allow-budget-shortfall --index results
"$py" paired_analysis.py --phase confirmation
"$py" paired_analysis.py --phase development_rescored
"$py" analyze_transfer_depth.py
"$py" result_tables.py
"$py" render_summary.py
"$py" - "$stamp" <<'PY'
import json
from pathlib import Path
import sys

coverage = json.loads(Path("results/CONFIRMATION_COVERAGE.json").read_text())
known = coverage["shards"]
planned = coverage["planned_shards"]
markdown = (
    "# 评测快照\n\n"
    f"服务器记录时间为 {sys.argv[1]}。训练已完成，共验证并归档 {known}/{planned} 个冻结确认单元。"
    "其余确认评测在服务器继续运行。快照发布仅包含当时已有完整回执的推理。\n\n"
    "- `REPORT_ZH.md` 记录当时可确认的主要结果。\n"
    "- `results/CONFIRMATION_COVERAGE.json` 列出尚未进入快照的条件。\n"
    "- `results/CONFIRMATION.csv` 含已可逐题配对的条件。`results/confirmation.tar.gz` 含全部已完成的逐题结果。\n"
    "- `GPU_LEDGER.tsv` 保留正在运行的作业开始记录，表中 GPU 用时以已结束作业为准。\n"
)
Path("SNAPSHOT.md").write_text(markdown, encoding="utf-8")
report = Path("REPORT_ZH.md")
lines = report.read_text(encoding="utf-8").split("\n", 1)
note = f"\n\n这是在 {sys.argv[1]} 归档的运行中快照，含 {known}/{planned} 个确认单元。 [快照范围](SNAPSHOT.md)\n"
report.write_text(lines[0] + note + lines[1], encoding="utf-8")
PY
bash package_release.sh --snapshot
printf 'Snapshot workspace %s\n' "$snap"
