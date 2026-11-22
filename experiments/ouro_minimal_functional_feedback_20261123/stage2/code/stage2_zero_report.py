"""Audit zero-shot and paper-shot results over identical question IDs."""
import json
from pathlib import Path
import sys

import numpy as np

from benchmark import holm, paired
from minimal import dump, load_jsonl, sha
from stage2_report import ARMS, BASELINES, dataset
from stage2_score import APPLE_BENCHMARKS


ALREADY_ZERO = {"sciq", "piqa"}


def status(rows, arm, metric):
    pred = []
    for item in rows:
        score = np.array(item["scores"][arm])
        if metric == "acc_norm":
            score = score / item["denominators"]
        pred.append(score.argmax() == item["gold"])
    return np.array(pred)


def run(root):
    zero = root / "zero_shot"
    summary = {}
    for name in APPLE_BENCHMARKS:
        reused = name in ALREADY_ZERO
        branch = root if reused else zero
        measure = dataset(branch, name)
        a_src = branch / "scores" / name / "RESULTS.jsonl"
        b_src = root / "scores" / name / "RESULTS.jsonl"
        lock = json.loads(a_src.with_name("LOCK.json").read_text())
        assert {row["shots"] for row in lock["harness_manifest"]["tasks"]} == {0}
        a, b = load_jsonl(a_src), load_jsonl(b_src)
        for x,y in zip(a,b,strict=True):
            assert (x["id"],x["gold"],x["choices"]) == (y["id"],y["gold"],y["choices"])
        for metric, value in measure["measurements"].items():
            value["paper_minus_zero_same_method"] = paired(
                status(a, "minimal_k4", metric), status(b, "minimal_k4", metric)
            )
        summary[name] = dict(measure, fewshot=0, reused_zero_in_paper_recipe=reused,
                             paper_score_sha256=sha(b_src))
    for baseline in BASELINES:
        holm([m["paired"][baseline] for item in summary.values() for m in item["measurements"].values()])
    dump(zero / "COMPARISON.json", summary)
    lines = [
        "# 同题目、同预算的七项 zero-shot 结果",
        "",
        "K4 复用已冻结的 32,768 参数线性反馈，对照沿用同评分器的原生、固定和自适应 LoopCD。"
        "保留论文设置中的 SciQ 和 PIQA 原始 0-shot 结果。另五项将示例数设为 0，保留数据修订和完整逐题集合。"
        "两套结果逐题核对了 ID、标准答案和候选文本。",
        "",
        "| 任务 | 题数 | 指标 | Native K1 | Native K4 | 固定 LoopCD K4 | 自适应 LoopCD K4 | 最简反馈 K4 | 主方法加示例后的差值 pp |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in summary.items():
        n = item["n"]
        for metric, m in item["measurements"].items():
            cells = " | ".join(f"{100*m['correct'][tag]/n:.2f}% ({m['correct'][tag]})" for tag in ARMS)
            difference = m["paper_minus_zero_same_method"]["delta_pp"]
            lines.append(f"| {key} | {n} | {metric} | {cells} | {difference:+.2f} |")
    lines += [
        "",
        "全部任务对 Native、固定 LoopCD 和自适应 LoopCD 的救回、退化、区间和 Holm p 值见 COMPARISON.json。",
        "SciQ 和 PIQA 的逐题文件链接回论文设置目录中的同一份结果。",
        " [SciQ](../scores/sciq/RESULTS.jsonl.gz) [PIQA](../scores/piqa/RESULTS.jsonl.gz)",
    ]
    (zero / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run(Path(sys.argv[1]))
