"""Summarize complete paired benchmark rows without treating a partial panel as full."""

import argparse
import json
from pathlib import Path
import statistics

from evaluate_transfer import TASKS


MODEL_LABELS = {"ouro-2.6b": "Ouro-2.6B", "ouro-1.4b-thinking": "Ouro-1.4B-Thinking",
                "ouro-2.6b-thinking": "Ouro-2.6B-Thinking"}


def format_arms(values, scale=1, suffix=""):
    return " / ".join(f"{value / scale:.2f}{suffix}" if value is not None else "—" for value in values)


def main(a):
    rows = ["# 冻结 T1 的 Ouro 跨版本评测", "",
            "使用同一历史 T1 权重，在 K4 的第三次归一化前按原接收位置写入，所有方法固定原生四轮。",
            "每个模型使用自己的 tokenizer，Native 与 T1 共享同模型的提示、解码和预算。",
            "前三个版本的实验无需训练。Huginn 使用独立流程。", "",
            "| 模型 | Benchmark | N | Native | T1 | 差值 pp [95%] | 救回 / 退化 | Native / T1 秒 |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- |"]
    closed = []
    sources = {}
    auxiliary = []
    interpretation = []
    if a.historical is not None:
        historic = json.loads(a.historical.read_text(encoding="utf-8"))
        old = {}
        for task, r in historic["results"].items():
            if task not in TASKS or r["metric"] != TASKS[task][1]:
                raise ValueError(f"Historical Base protocol differs for {task}")
            metrics = {}
            for name, p in r["paired"]["native"].items():
                metrics[name] = dict(n=p["n"], native=p["reference"], t1=p["controller"], delta=p["delta"],
                                     paired_ci95=p["paired_ci95"], rescue=p["rescued"], damage=p["harmed"])
            timings = dict(native=r.get("baseline_cost", {}).get("native", {}), t1=r["t1_cost"])
            old[task] = dict(primary=r["metric"], metrics=metrics, cost=timings)
        sources["Ouro-1.4B Base（历史）"] = old
    for key, label in MODEL_LABELS.items():
        target = a.results / key / "RESULTS.json"
        if not target.is_file():
            continue
        sources[label] = json.loads(target.read_text(encoding="utf-8"))
    for label, result in sources.items():
        macro = {}
        for task in TASKS:
            if task not in result:
                continue
            r = result[task]
            pair = r["metrics"][r["primary"]]
            if pair["n"] != TASKS[task][0]:
                raise ValueError(f"Wrong full population {label}/{task}")
            macro[task] = pair
            costs = (r["cost"]["native"], r["cost"]["t1"])
            timings = [cost.get("elapsed_seconds", cost.get("seconds")) for cost in costs]
            low, high = pair["paired_ci95"]
            values = [label, task, f'{pair["n"]:,}', f'{100*pair["native"]:.2f}%', f'{100*pair["t1"]:.2f}%',
                      f'{100*pair["delta"]:+.2f} [{100*low:+.2f}, {100*high:+.2f}]',
                      f'{pair["rescue"]} / {pair["damage"]}', format_arms(timings)]
            rows.append("| " + " | ".join(values) + " |")
            keys = ("parseable", "strict_format", "strict") if task == "mmlu_pro" else ("acc", "acc_norm")
            metrics = [
                f"{key} {format_arms([100 * r['metrics'][key][arm] for arm in ('native', 't1')], suffix='%')}"
                for key in keys if key in r["metrics"]
            ]
            allocated = format_arms([cost.get("peak_allocated_bytes") for cost in costs], 2**30)
            reserved = format_arms([cost.get("peak_reserved_bytes") for cost in costs], 2**30)
            auxiliary.append(f"| {label} | {task} | {'，'.join(metrics) or '—'} | {allocated} | {reserved} |")
        if len(macro) == len(TASKS):
            if label in MODEL_LABELS.values():
                closed.append(label)
                positive = [task for task, result in macro.items() if result["paired_ci95"][0] > 0]
                negative = [task for task, result in macro.items() if result["paired_ci95"][1] < 0]
                interpretation.append(
                    f"- {label} 的逐题配对区间全为正的任务有 {', '.join(positive) or '无'}，"
                    f"区间全为负的任务有 {', '.join(negative) or '无'}。"
                )
            values = [100 * statistics.mean(r[field] for r in macro.values()) for field in ("native", "t1", "delta")]
            rows.append(f"| {label} | 八项等权平均 | 43,769 | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:+.2f} |  |  |")
    if a.final and len(closed) != len(MODEL_LABELS):
        raise ValueError("The three Ouro models have not completed all eight benchmarks")
    rows.extend(["", f"已完成八项全量对比的 Ouro 版本数为 {len(closed)}/{len(MODEL_LABELS)}。",
                 "", "## 附加指标和计算成本", "",
                 "每格依次为 Native 和 T1。时间为本次运行在对应设备上各评测分片的实测耗时之和。"
                 "显存为 PyTorch 记录的峰值，历史未记录的数值用横线表示。",
                 "", "| 模型 | Benchmark | 附加指标 | 分配峰值 GiB | 保留峰值 GiB |",
                 "| --- | --- | --- | --- | --- |", *auxiliary, "",
                 "## 跨版本证据", "", *interpretation, "",
                 "配对区间按题重采样，各任务单独计算，未校正多重比较。"
                 "冻结权重在同一提示和循环预算下改变了性能，这一对比检验直接权重迁移。"
                 "确认共享功能坐标及分离直接读出和原生循环的贡献需要独立对照。", ""])
    a.out.write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--final", action="store_true")
    main(parser.parse_args())
