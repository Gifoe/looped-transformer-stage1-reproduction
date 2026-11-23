"""Write reproducible Chinese evidence tables from immutable confirm receipts."""

import json
from pathlib import Path


R = Path("results")
ARC = R/"HOLDOUT/arc/depth/s261123137/confirm"
POS = R/"HOLDOUT/arc/position/s261123137/confirm"
CSQA = R/"HOLDOUT/csqa/source/s261123137/confirm"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, body):
    Path(path).write_text("\n".join(body)+"\n", encoding="utf-8")


def interval(x, scale=1.):
    return "[" + ", ".join(f"{scale*v:+.3f}" for v in x) + "]"


def accuracy(x):
    y=x["statistics"]["acc_norm"]
    return f'{y["controlled"]} ({y["mean_gain"]:+.2f} pp)'


def depth_report():
    arc = read(ARC/"SUMMARY.json")["evaluation"]["arc_confirm"]
    stats = arc["results"]
    lines = ["# 真正的深度留出", "",
             "ARC-Challenge Train 1,119 题，冻结 Ouro，所有组共享实际初始化、每题候选归一及 282 次更新。只改变训练时执行的深度。评估为与训练、发现和已核验历史评分互斥的 ARC-Easy Validation 299 题。每格为 acc_norm 正确数及相对同深度 Native 的百分点。", "",
             "| 训练深度 | K2 | K3 | K4 | K5 | K6 |", "|---|---:|---:|---:|---:|---:|"]
    baseline = [stats[f"k2_k{k}"]["statistics"]["acc_norm"]["native"] for k in range(2,7)]
    lines.append("| Native | " + " | ".join(str(x) for x in baseline) + " |")
    for group in ("k2", "k3", "k4", "k23", "k24", "k234"):
        lines.append(f"| {group.upper()} | " + " | ".join(accuracy(stats[f"{group}_k{k}"]) for k in range(2, 7)) + " |")
    lines += ["", "所有原始分数、救回和退化、按题置信区间、候选长度、原始 acc、实际写入剂量及耗时均在同一确认目录。",
              f"[完整深度配对]({ARC}/SUMMARY.json) · [逐题分数]({ARC}/arc_confirm_PREDICTIONS.jsonl)", "",
              "## 固定 K4，专门训练不同注入位置", "",
              "| 训练位置 | 在第 1 圈后写入 | 在第 2 圈后写入 | 在第 3 圈后写入 |",
              "|---|---:|---:|---:|"]
    pos = read(POS/"SUMMARY.json")["evaluation"]["arc_confirm"]["results"]
    for trained in range(1, 4):
        lines.append(f"| 第 {trained} 圈后 | " + " | ".join(accuracy(pos[f"p{trained}_k4_p{p}"]) for p in range(1, 4)) + " |")
    lines += ["", "早写专门训练后同样可以产生明确收益。训练条件会改变位置排名。以这九格的点估计，第 3 圈专门训练后在本位置最高，但早写组在第 1、2 圈仍接近。两者差异须看同题成对差，不能由各自相对原生的区间相减。",
              f"[九格候选配对]({POS}/SUMMARY.json)"]
    save("DEPTH_TRANSFER.md", lines)


def geometry_report():
    folder = R/"state_horizon/arc_confirm"
    doc = read(folder/"STATE_HORIZON.json")
    conditions = doc["conditions"]
    lines = ["# 实际 RMSNorm 和状态 × 时域响应", "",
             "固定历史 Rank-1，在 96 道事前哈希抽取的独立确认题上交叉三处真实 pre-RMSNorm 状态和 0–3 圈后缀。候选及各原生预算相同。图 1 显示实际分解，功能谱对照只使用 Train。", "",
             "![状态与后缀长的效用](results/PREDICTION_AUDIT/FIGURE_1_STATE_HORIZON.png)", "",
             "| 接收状态 | 后缀圈数 | 已学更新 Δq | 切向 Δq | 径向 Δq | 有限公式/真实向量余弦中位数 | 线性/真实余弦中位数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, row in conditions.items():
        k, m = key.split("_")
        r = row["result"]
        g = r["learned"]["geometry_medians"]
        lines.append(f"| {k} | {m[1:]} | {r['learned']['utility']['delta']:+.4f} | "
                     f"{r['tangent_component']['utility']['delta']:+.4f} | {r['radial_component']['utility']['delta']:+.4f} | "
                     f"{g['cosine_exact']:.4f} | {g['cosine_linear']:.4f} |")
    lines += ["", "实际射影与解析结果均加入 Ouro 的 learned RMSNorm 权重、epsilon 和 BF16 写入舍入。径向和切向作为各自独立反事实，二者效用无需相加。后归一化等相对限幅的干预改变真实位置及可用语义，单列完整结果。自然位移使用下一圈的原生状态，仅作研究对照。",
              f"[逐条件指标和分片哈希]({folder}/STATE_HORIZON.json) · [可编辑图 1](results/PREDICTION_AUDIT/FIGURE_1_STATE_HORIZON.pdf)"]
    save("NORMALIZED_GEOMETRY.md", lines)


def transport_report():
    folder = R/"D_TRANSPORT/arc_confirm"
    doc = read(folder/"TRANSPORT_SUMMARY.json")
    lines = ["# 冻结后缀上的早写与晚写", "",
             "在 299 道独立 ARC-Easy 上，将第 1 或第 2 圈后的早写传播至下一边界。晚写直接在同一接收状态执行。按每个候选的整条有效序列匹配实际 BF16 写回 RMS，再用同一个原生后缀评分。对照两种原始位置、实际等剂量、Attention 或 MLP 恢复及同计算路径。", "",
             "| 边界 | 干预 | Δq | 95% 配对区间 | 梯度点积与 Δq r | 范数与 Δq r |",
             "|---|---|---:|---:|---:|---:|"]
    for name, item in doc["results"].items():
        for arm, outcome in item["conditions"].items():
            q = outcome["metrics"]["utility"]
            relationship = outcome["gradient_vs_observed"]
            pearson = relationship["pearson"]
            norm = relationship["norm_pearson"]
            a, b = ("–" if n is None else f"{n:.3f}" for n in (pearson, norm))
            lines.append(f"| {name} | {arm} | {q['delta']:+.4f} | {interval(q['ci95'])} | {a} | {b} |")
        lines += ["", f"**{name} 的直接成对比较**。每个候选都符合剂量匹配容差的题数为 "
                  f"{item['questions_with_all_candidates_dose_matched']}/{doc['n']}。", ""]
        for a, x in item["contrasts"].items():
            y = x["utility"]
            lines.append(f"- `{a}` 的 Δq 为 {y['delta']:+.4f}，95% 区间 {interval(y['ci95'])}。")
        lines += ["", "| 边界 | 干预 | Δq | 95% 配对区间 | 梯度点积与 Δq r | 范数与 Δq r |",
                  "|---|---|---:|---:|---:|---:|"]
    lines = lines[:-2]
    lines += ["", "梯度点积与范数的相关系数是同题统计。恢复 Attention 或 MLP 后重新执行后续模块会包含路径交互，故不能根据该数据给两者作互斥贡献百分比。完整有效序列逐 token 传播保留于 chunks 下。 ",
              f"[配对和实际计算成本]({folder}/TRANSPORT_SUMMARY.json) · [候选原始分数]({folder}/PREDICTIONS.jsonl)"]
    save("CONTROL_TRANSPORT.md", lines)


def transfer_report():
    arc, csqa = read(ARC/"csqa_transfer/SUMMARY.json")["evaluation"], read(CSQA/"SUMMARY.json")["evaluation"]
    subsets = (("ARC→CSQA", arc["csqa_confirm"]), ("CSQA→ARC-Easy", csqa["arc_confirm"]),
               ("CSQA→CSQA", csqa["csqa_confirm"]), ("ARC→ARC-Easy", read(ARC/"SUMMARY.json")["evaluation"]["arc_confirm"]))
    lines = ["# 跨任务迁移和独立训练来源比较", "",
             "第一轮两种训练来源都从零开始，各使用 1,119 道官方训练题，完全相同的更新次数和初始化。两种测量域各 299 道确认题。ARC 训练使用 ARC-Challenge，ARC 侧确认使用独立 ARC-Easy。以下为逐深度 acc_norm，括号中的区间和救回/退化均按同题 Native。", "",
             "| 路径 | 深度 | Native→控制 | Δpp | 95% CI | 救回/退化 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, subset in subsets:
        for k in range(2, 7):
            row = subset["results"][f"k234_k{k}"]["statistics"]["acc_norm"]
            lines.append(f"| {name} | K{k} | {row['native']}→{row['controlled']}/{subset['n']} | "
                         f"{row['mean_gain']:+.2f} | {interval(row['ci95'])} | {row['rescued']}/{row['harmed']} |")
    if (R/"ROUND2_AUDIT/RESULTS.json").is_file():
        lines += ["", "## 第二轮五种训练配方，两个随机种子", "",
                  "新增 OpenBookQA、QASC、ARC/CSQA 均衡混合。评估为与第一轮确认和所有训练源互斥的 "
                  "70 道 ARC-Easy 和 299 道 CSQA。每格是两个种子、K2–K6 的 acc_norm 平均净百分点。",
                  "", "| 训练来源 | ARC-Easy | CSQA | 两域中较小收益 |", "|---|---:|---:|---:|"]
        for row in read(R/"ROUND2_AUDIT/RESULTS.json")["source_aggregate"]:
            lines.append(f"| {row['source']} | {row['arc_avg_acc_norm_pp']:+.2f} | {row['csqa_avg_acc_norm_pp']:+.2f} | {row['maximin_acc_norm_pp']:+.2f} |")
        lines += ["", "[第二轮各来源、种子、深度的完整配对](results/ROUND2_AUDIT/RESULTS.json) · "
                  "[事前预测锁](results/forecast_round2/FORECAST_LOCK.json)"]
    lines += ["", f"[ARC 训练→CSQA 原始配对]({ARC}/csqa_transfer/SUMMARY.json) · [CSQA 训练原始配对]({CSQA}/SUMMARY.json) · "
              "[官方划分及独立性限制](DATASET_SELECTION.md)"]
    save("CROSS_TASK_TRANSFER.md", lines)


if __name__ == "__main__":
    depth_report()
    geometry_report()
    transport_report()
    transfer_report()
