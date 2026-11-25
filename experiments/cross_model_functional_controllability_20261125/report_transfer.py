"""Summarize complete paired benchmark rows without treating a partial panel as full."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from evaluate_transfer import TASKS


MODEL_LABELS = {"ouro-2.6b": "Ouro-2.6B", "ouro-1.4b-thinking": "Ouro-1.4B-Thinking",
                "ouro-2.6b-thinking": "Ouro-2.6B-Thinking"}
HUGINN_LABEL = "Huginn-0125（K32）"


def format_arms(values, scale=1, suffix=""):
    return " / ".join(f"{value / scale:.2f}{suffix}" if value is not None else "—" for value in values)


def generation_transitions(directory):
    def metrics(arm):
        with (directory / f"mmlu_pro__{arm}.jsonl").open(encoding="utf-8") as stream:
            return {row["id"]: row["metrics"] for line in stream if (row := json.loads(line))}

    native, t1 = metrics("native"), metrics("t1")
    if native.keys() != t1.keys() or len(native) != TASKS["mmlu_pro"][0]:
        raise ValueError("Missing paired generation populations")
    counts = {}
    for key, before in native.items():
        after = t1[key]
        group = (before["parseable"], after["parseable"])
        stats = counts.setdefault(group, Counter())
        stats.update(n=1, native=before["correct"], t1=after["correct"])
    return counts


def functional_rows(folder):
    rows = []
    ouro = folder / "mechanism" / "MECHANISM.json"
    if ouro.is_file():
        measured = json.loads(ouro.read_text(encoding="utf-8"))
        for slug, spectra in measured["spectra"].items():
            label = MODEL_LABELS.get(slug, "Ouro-1.4B Base")
            dose = next(r for r in measured["finite"][slug] if r["dose"] == 1 and r["sign"] == "positive")
            direct, loop, contrast = dose["mean"]
            rows.append(f"| {label} | K4 | {spectra['ordinary']['stable_rank']:.2f} | "
                        f"{spectra['contrast']['stable_rank']:.2f} | {direct:+.4f} | {loop:+.4f} | {contrast:+.4f} |")
    huginn = folder / "huginn-0125" / "mechanism" / "MECHANISM.json"
    if huginn.is_file():
        measured = json.loads(huginn.read_text(encoding="utf-8"))
        spectra = measured["spectra"]
        dose = next(r for r in measured["finite"] if r["dose"] == 1 and r["sign"] == 1)
        direct, loop, contrast = dose["means"]
        rows.append(f"| Huginn-0125 | K32 | {spectra['ordinary']['stable_rank']:.2f} | "
                    f"{spectra['contrast']['stable_rank']:.2f} | {direct:+.4f} | {loop:+.4f} | {contrast:+.4f} |")
    return rows


def main(a):
    title = "Ouro 与 Huginn 的低秩功能控制" if a.require_huginn else "冻结 T1 的 Ouro 跨版本评测"
    rows = [f"# {title}", "",
            "使用同一历史 T1 权重，在 K4 的第三次归一化前按原接收位置写入。每个 Ouro 对照执行四轮。",
            "每个模型使用自己的 tokenizer，Native 与 T1 共享同模型的提示、解码和预算。",
            "三个 Ouro 迁移实验使用冻结的历史 T1。Huginn K32 使用在 ARC-Challenge Train 单独拟合的同结构 Rank-1。",
            "所有模型在循环入口的预测 token 位置应用同一条相对 RMS 限幅规则。"
            r"设 $r=u(w^\top h)$，则 $h'=h+r\min(1,0.45\operatorname{RMS}(h)/\operatorname{RMS}(r))$，"
            "零更新保持为零。循环次数与延迟按各自架构报告。", "",
            "| 模型 | Benchmark | N | Native | Rank-1 | 差值 pp [95%] | 救回 / 退化 | Native / Rank-1 秒 |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- |"]
    closed = []
    sources = {}
    folders = {}
    auxiliary = []
    interpretation = []
    generation = []
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
    labels = {**MODEL_LABELS, "huginn-0125": HUGINN_LABEL}
    for key, label in labels.items():
        target = a.results / key / "RESULTS.json"
        if not target.is_file():
            continue
        sources[label] = json.loads(target.read_text(encoding="utf-8"))
        folders[label] = target.parent
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
            if label in labels.values():
                closed.append(label)
                positive = [task for task, result in macro.items() if result["paired_ci95"][0] > 0]
                negative = [task for task, result in macro.items() if result["paired_ci95"][1] < 0]
                interpretation.append(
                    f"- {label} 的逐题配对区间全为正的任务有 {', '.join(positive) or '无'}，"
                    f"区间全为负的任务有 {', '.join(negative) or '无'}。"
                )
                for (before, after), counts in sorted(generation_transitions(folders[label]).items()):
                    status = f"{'合法' if before else '非法'} → {'合法' if after else '非法'}"
                    generation.append(f"| {label} | {status} | {counts['n']:,} | {counts['native']:,} | "
                                      f"{counts['t1']:,} | {counts['t1']-counts['native']:+,} |")
            values = [100 * statistics.mean(r[field] for r in macro.values()) for field in ("native", "t1", "delta")]
            rows.append(f"| {label} | 八项等权平均 | 43,769 | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:+.2f} |  |  |")
    n_ouro = sum(label in closed for label in MODEL_LABELS.values())
    if a.final and n_ouro != len(MODEL_LABELS):
        raise ValueError("The three Ouro models have not completed all eight benchmarks")
    if a.require_huginn and HUGINN_LABEL not in closed:
        raise ValueError("Huginn has not completed the eight-benchmark comparison")
    rows.extend(["", f"已完成八项全量对比的 Ouro 版本数为 {n_ouro}/{len(MODEL_LABELS)}。",
                 f"Huginn 全量对比{'已完成' if HUGINN_LABEL in closed else '待完成'}。",
                 "", "## 附加指标和计算成本", "",
                 "每格依次为 Native 和 Rank-1。时间为本次运行在对应设备上各评测分片的实测耗时之和。"
                 "显存为 PyTorch 记录的峰值，历史未记录的数值用横线表示。",
                 "", "| 模型 | Benchmark | 附加指标 | 分配峰值 GiB | 保留峰值 GiB |",
                 "| --- | --- | --- | --- | --- |", *auxiliary, "",
                 "## MMLU-Pro 生成答案分解", "",
                 "用原始解析器配对划分每道题的合法输出状态。净正确数为该组中 Rank-1 与 Native 的差，"
                 "每组题目由两个实际输出共同定义。",
                 "", "| 模型 | 合法状态 | 题数 | Native 正确 | Rank-1 正确 | 净正确数 |",
                 "| --- | --- | ---: | ---: | ---: | ---: |", *generation, "",
                 "## 跨版本证据", "", *interpretation, "",
                 "配对区间按题重采样，各任务单独计算，未校正多重比较。"
                 "Ouro 各版本在同一提示和循环预算下比较冻结 T1，以检验直接权重迁移。"
                 "共享功能坐标以及原生循环的贡献由独立功能响应对照评估。", ""])
    measured = functional_rows(a.results)
    if measured:
        rows.extend(["## 功能矩阵与实际干预", "",
                     r"令 $H$ 为完整序列在写入边界的状态，$F$ 为最后一轮原生计算，$q$ 为继续读出的任务效用。",
                     r"$M_D=\mathbb{E}_D\sum_t\nabla_{H_t}(q\circ F)(H)\,H_t^\top$。",
                     r"$C_D=M_D-\mathbb{E}_D\sum_t\nabla_{H_t}q(H)\,H_t^\top$。",
                     r"求和覆盖与控制器一致的预测位置，导数经过完整序列，因此包含跨位置影响。",
                     r"无限小、尚未触及限幅的 Rank-1 写入的循环净效用近似为 $\epsilon u^\top C_Dw$。",
                     "有限剂量质量由原生继续执行后的实际响应检验。直接对照在同一写入边界调用模型读出。"
                     "Huginn 的直接对照保留原生 Coda，配对条件共享随机初态和输入注入。",
                     "",
                     "谱在 ARC-Challenge Train 全量 1,119 题上估计。有限剂量对照使用独立的 24 道验证题，"
                     "表中效用为字符归一化金选项相对其余候选的对数间隔。效用变化与测试准确率使用各自的单位。",
                     "", "| 模型 | 深度 | M 稳定秩 | C 稳定秩 | 直接效用差 | 继续一轮效用差 | 两者差 |",
                     "| --- | --- | ---: | ---: | ---: | ---: | ---: |", *measured, ""])
    a.out.write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--require-huginn", action="store_true")
    main(parser.parse_args())
