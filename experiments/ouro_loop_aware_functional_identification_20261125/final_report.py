"""Produce the Chinese report from completed experiment receipts."""

import argparse
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def full_dose(root):
    study = read(root / "AUDIT.json")
    return next(row for row in study["confirmation"]
                if row["depth"] == 4 and row["dose"] == 1 and row["scope"] == "leave_source_out")


def mean_for(rows, method):
    return statistics.mean(row["paired"][method][metric]["baseline"] for _, metric, row in rows)


def dev_table(root):
    choices = {
        "dev": ("uniform", "ordinary", "loop", "warm_uniform", "warm_loop"),
        "dev_refinement": ("warm_tangent",),
        "dev_within_source": tuple(f"within_{source}_{mode}" for source in ("arc_challenge", "csqa")
                                   for mode in ("uniform", "high", "low")),
        "dev_coverage": ("coverage_uniform", "coverage_tangent", "coverage_loss_tangent"),
    }
    endpoints = ("arc_challenge/candidate", "arc_challenge/content", "csqa/candidate",
                 "mmlu/mmlu", "gsm8k/gsm_process")
    lines = ["| 训练对照 | ARC 候选 | ARC 自由生成 | CSQA 候选 | MMLU | GSM8K 严格 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for folder, methods in choices.items():
        experiments = read(root / "results" / folder / "SELECTION.json")["summary"]
        for name in methods:
            row = experiments[name]["gap"]
            values = [f"{100*row[k]:+.2f}" for k in endpoints]
            lines.append("| " + " | ".join([name, *values]) + " |")
    return lines


def assemble(root):
    official = root / "results/official"
    result = read(official / "RESULTS.json")
    selection = read(root / "results/final_development/SELECTION.json")
    fit = read(root / "results/fits" / selection["best"] / "TRAIN.json")
    check = read(official / "FREEZE.json")
    if check["arms"]["new"]["sha256"] != selection["checkpoint_sha256"]:
        raise ValueError("Final test and development checkpoint differ")
    fields = result["primary"]
    available = {}
    for row in result["tasks"]:
        task = row["task"]
        if task not in available and fields[task] in row["paired"]["native"]:
            available[task] = row
    primary = [(task, key, available[task]) for task, key in fields.items() if task in available]
    if len(primary) != 12:
        raise ValueError("The Chinese report requires the twelve official endpoints")
    original = {"arc_challenge", "arc_easy", "sciq", "mmlu", "hellaswag", "winogrande", "piqa", "mmlu_pro"}
    legacy = [row for row in primary if row[0] in original]
    new_score = statistics.mean(row["paired"]["native"][metric]["new"] for _, metric, row in legacy)
    gap_t1 = new_score - mean_for(legacy, "t1")
    gap_rank8 = new_score - mean_for(legacy, "rank8")
    mmlu = available["mmlu"]["paired"]["native"]["acc"]
    mmlu_pro = available["mmlu_pro"]["paired"]["native"]
    six = root / "results/forecast_candidate"
    ten = root / "results/forecast_all_sources"
    evidence = full_dose(six)["methods"]
    extrapolation = full_dose(ten)["methods"]
    pretrained = read(six / "TRAINED_VALUE.json")
    prospect = pretrained[0]["prospective_training_rewards"]["leave_source_out"]["methods"]
    trajectory = read(ten / "FORECAST.json")

    policy = selection["best"]
    fit_sources = "、".join(fit["sources"])
    task_groups = "、".join(trajectory["source_order"])
    hours = sum(row["costs"]["new"].get("seconds", row["costs"]["new"].get("elapsed_seconds", 0.))
                for _, _, row in primary) / 3600
    lower = [task for task, key, row in primary if row["paired"]["native"][key]["change"] < 0.]
    weak = "、".join(lower) if lower else "无"
    lines = [
        "# Loop-Aware Functional Identification 完整研究报告",
        "",
        f"独立开发选出的模型为 `{policy}`。最终权重 SHA-256 为 `{selection['checkpoint_sha256']}`。"
        f"模型使用 {fit_sources}，累计 {fit['items']:,} 题、{fit['updates']:,} 次更新和 "
        f"{fit['padded_token_loops']:,} 个填充后 token-loop。Ouro-1.4B Base 权重冻结，全部正式评测使用同一 K4 检查点。",
        "",
        f"前八项严格同协议的等权平均为 **{100*new_score:.2f}%**。"
        f"相对 T1 为 **{100*gap_t1:+.2f}** 个百分点，相对 Rank-8 为 **{100*gap_rank8:+.2f}** 个百分点。"
        f"MMLU 相对 Native 变化 **{100*mmlu['change']:+.2f}** 个百分点，"
        f"95% 区间为 [{100*mmlu['ci95'][0]:+.2f}, {100*mmlu['ci95'][1]:+.2f}]。"
        f"MMLU-Pro 的答题正确率和合法输出率相对 Native 分别变化 "
        f"{100*mmlu_pro['correct']['change']:+.2f} 和 {100*mmlu_pro['parseable']['change']:+.2f} 个百分点。",
        f"新模型十二个主任务实际计时合计为 {hours:.2f} 小时。相对 Native 主指标下降的任务为 {weak}。",
        "",
        "[十二项结果表](results/official/RESULTS.md) · [逐项配对区间](results/official/RESULTS.csv) · "
        "[预测、生成诊断及计算量回执](results/official/RESULTS.json)",
        "",
        "## 循环效用与来源选择",
        "",
        "固定同一前 RMSNorm 接收状态，使用完整序列回传，比较直接读出与一次完整原生循环。"
        "实验分别测量可跨词元传播的局部梯度、实际限幅剂量、整答正确率和训练控制收益。",
        "",
        f"十来源机制池为 {task_groups}。六来源的独立重放在 K4 全剂量下，"
        f"普通功能矩阵与循环交互矩阵的平均留一来源 Spearman 分别为 "
        f"{evidence['ordinary_functional']['aggregate']['spearman_mean']:.3f} 和 "
        f"{evidence['loop_contrast']['aggregate']['spearman_mean']:.3f}。"
        f"扩大到十来源后，结果为 {extrapolation['ordinary_functional']['aggregate']['spearman_mean']:.3f} "
        f"和 {extrapolation['loop_contrast']['aggregate']['spearman_mean']:.3f}。"
        "教师强制生成的词元指标及自由生成正确率分别报告。",
        "",
        f"实际等题数单源训练跨独立开发环境的普通投影与 T1 切空间评分，"
        f"平均选型后悔值分别为 {prospect['ordinary_functional']['aggregate']['regret_mean']:.4f} "
        f"与 {prospect['loop_factor_influence_warm_t1']['aggregate']['regret_mean']:.4f}。"
        "后悔值比较对应开发环境上所选来源与该对照中观测最优来源的真实准确率差。",
        "",
        "## 配对训练对照",
        "",
        "下表为独立开发集上相对历史 T1 的百分点。ARC 与 CSQA 的候选评分与实际生成分开。"
        "主训练五组使用相同题目、顺序、训练步数和 token-loop，区别在样本权重或初始化。"
        "同源高低权重对照在每个原生正确转移及题长组内保持总权重。",
        "",
        *dev_table(root),
        "",
        "## 机制边界",
        "",
        "ARC 上存在局部有效的循环方向，但 K2 与 K4 的首方向余弦只有 0.224。"
        "BoolQ 的稳定主轴表现为 Yes 偏差，按训练损失对齐后效应符号由正变负。"
        "因此源域低秩、方向稳定或较大的交互项单独都不足以推断跨任务收益。"
        "T1 与 Mixed 的历史差异还混有训练来源与优化过程，归因应以本轮匹配对照为准。",
        "",
        "[完整数学推导](THEORY.md) · [数据价值预测](DATA_SELECTION.md) · [因果机制与失败分析](MECHANISM.md) · [单卡复现](REPRODUCE.md)",
        "",
        "![正式评测的配对性能变化](figures/official/official_k4_performance.svg)",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    a.out.write_text(assemble(a.root), encoding="utf-8")
