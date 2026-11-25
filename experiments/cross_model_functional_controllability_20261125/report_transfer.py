"""Summarize complete paired benchmark rows without treating a partial panel as full."""

import argparse
import json
from pathlib import Path
import statistics

from evaluate_transfer import TASKS


MODEL_LABELS = {"ouro-2.6b": "Ouro-2.6B", "ouro-1.4b-thinking": "Ouro-1.4B-Thinking",
                "ouro-2.6b-thinking": "Ouro-2.6B-Thinking"}


def main(a):
    rows = ["# 冻结 T1 的 Ouro 跨版本评测", "",
            "使用同一历史 T1 权重，在 K4 的第三次归一化前按原接收位置写入，所有方法固定原生四轮。",
            "每个模型使用自己的 tokenizer，Native 与 T1 共享同模型的提示、解码和预算。",
            "前三个版本的实验无需训练。Huginn 使用独立流程。", "",
            "| 模型 | Benchmark | N | Native | T1 | 差值 pp [95%] | 救回 / 退化 | Native / T1 秒 |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- |"]
    closed = []
    for key, label in MODEL_LABELS.items():
        target = a.results / key / "RESULTS.json"
        if not target.is_file():
            continue
        result = json.loads(target.read_text(encoding="utf-8"))
        macro = []
        for task in TASKS:
            if task not in result:
                continue
            r = result[task]
            pair = r["metrics"][r["primary"]]
            if pair["n"] != TASKS[task][0]:
                raise ValueError(f"Wrong full population {label}/{task}")
            macro.append(pair)
            timings = [c.get("elapsed_seconds", c.get("seconds"))
                       for c in (r["cost"]["native"], r["cost"]["t1"])]
            low, high = pair["paired_ci95"]
            values = [label, task, f'{pair["n"]:,}', f'{100*pair["native"]:.2f}%', f'{100*pair["t1"]:.2f}%',
                      f'{100*pair["delta"]:+.2f} [{100*low:+.2f}, {100*high:+.2f}]',
                      f'{pair["rescue"]} / {pair["damage"]}', " / ".join(f"{val:.1f}" for val in timings)]
            rows.append("| " + " | ".join(values) + " |")
        if len(macro) == len(TASKS):
            closed.append(label)
            values = [100 * statistics.mean(r[field] for r in macro) for field in ("native", "t1", "delta")]
            rows.append(f"| {label} | 八项等权平均 | 43,769 | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:+.2f} |  |  |")
    if a.final and len(closed) != len(MODEL_LABELS):
        raise ValueError("The three Ouro models have not completed all eight benchmarks")
    rows.extend(["", f"已完成八项全量对比的 Ouro 版本数为 {len(closed)}/{len(MODEL_LABELS)}。"
                 "各模型的逐题分数、两项选项指标、MMLU-Pro 合法率与严格格式、显存峰值均随对应结果归档。", ""])
    a.out.write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    main(parser.parse_args())
