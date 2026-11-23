"""Render the comparison and training ablation tables from completed receipts."""

import json
from pathlib import Path

from freeze_eval import ENDPOINTS


METRIC_NAMES = {"acc_norm": "候选 acc_norm", "correct": "生成正确率", "strict": "严格数值",
                "process_correct": "过程与数值联合正确", "premature_eos": "标记前 EOS"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def percent(x):
    return f"{100 * x:.2f}%"


def difference(x):
    low, high = x["paired_ci95"]
    return f"{100*x['delta']:+.2f} [{100*low:.2f}, {100*high:.2f}]"


def table(headers, values):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers),
                      *("| " + " | ".join(map(str, row)) + " |" for row in values)])


def training_tables(root, config, dev, destination=Path("ABLATIONS.md"), coverage_note=""):
    run = read(root / "TRAINING_AUDIT.json")
    rows = [(name, data["updates"], data["unique_ids"], data["policy_prefixes"], f"{data['seconds']:.1f}",
             f"{data['padded_token_loops']:,}", f"{data['product_norm']:.4f}") for name, data in run["runs"].items()]
    latest = [(arm, f"{100*d['worst_delta']:+.2f}", f"{100*d['mean_delta']:+.2f}") for arm, d in config["validation"].items()]
    prefix = []
    for k in (2, 3, 4):
        phase = dev[f"gsm8k/k{k}/gsm_process"]
        on = phase["on_policy_resume"]
        teacher = phase["teacher_resume"]
        for metric in ("strict", "process_correct", "premature_eos"):
            detail = on["paired"]["teacher_resume"][metric]
            prefix.append((f"K{k}", METRIC_NAMES[metric], detail["n"], percent(teacher["metrics"][metric]["value"]),
                           percent(on["metrics"][metric]["value"]), difference(detail), f"{detail['rescued']}/{detail['harmed']}"))
    ablations = ["# 训练与消融", "",
                 "所有 4,096 参数控制器共用同一训练划分、冻结主干和轮换预算。原始排序只用可应用的候选实例。",
                 "", table(("控制器", "更新", "独立题号", "实际自由前缀数", "训练秒", "填充 token×loop", "内外因子范数乘积"), rows),
                 "", "## 开发集选型", "", "表中差值以每项开发指标的预先指定对照中最高者为参照，单位为百分点。", coverage_note,
                 "", table(("控制器", "最弱项差值", "平均差值"), latest),
                 "", "## 固定训练顺序的前缀对照", "",
                 "两次继续训练从同一权重出发，使用完全相同的 336 步、数据 ID、K2–K4 调度和学习率。自由前缀更新仅在训练可见。标记前 EOS 较低为好。",
                 "", table(("深度", "指标", "n", "参考前缀续训", "自由前缀续训", "配对差及区间 pp", "新命中／丢失命中"), prefix),
                 "", "实际训练轨迹、梯度与最终组权重见 results/TRAINING_AUDIT.json。开发全部逐题统计见 results/PAIRED_DEVELOPMENT.csv。", ""]
    destination.write_text("\n".join(ablations), encoding="utf-8")


def evaluation_tables(root, config, results):
    selected = config["selected"]
    primary = []
    for ds, mode, metric in ENDPOINTS:
        if f"{ds}/k4/{mode}" not in results:
            continue
        cell = results[f"{ds}/k4/{mode}"]
        if selected not in cell or "native" not in cell:
            continue
        own = cell[selected]
        before = own["paired"]["native"][metric]
        ref = config["reference_envelope"][f"{ds}/k4/{mode}/{metric}"]["arm"]
        matching = cell.get(ref)
        old_cost = 1e3 * cell["native"]["cost"]["seconds"] / before["n"]
        new_cost = 1e3 * own["cost"]["seconds"] / before["n"]
        primary.append((ds, METRIC_NAMES[metric], before["n"], percent(before["reference"]), percent(before["controller"]),
                        difference(before), f"{before['rescued']}/{before['harmed']}", ref,
                        percent(matching["metrics"][metric]["value"]) if matching else "—", f"{old_cost:.1f} / {new_cost:.1f}"))
    same = read(root / "MATCHED_DEPTH.json")
    transfer = []
    for ds, mode, metric in ENDPOINTS:
        if f"{ds}/{mode}" not in same:
            continue
        row = same[f"{ds}/{mode}"]
        values = [difference(row["depths"][f"k{k}"]["native"][metric]) for k in range(2, 7)]
        transfer.append((ds, METRIC_NAMES[metric], len(row["ids"]), *values))
    coverage = read(root / "PAIRED_CONFIRMATION.json")["coverage"]
    lead = ["# 正式确认结果", "", f"开发权重锁定为 {selected}。已验证 {coverage['cells']} / {coverage['expected']} 个确认条件。",
            "独立确认以完整词表实际输出为主。过程联合端点要求结构化过程和正确的严格最终数字。程序化过程格式不校验数学证明本身。",
            "", "## K4 主测评", "",
            "配对差值的区间为 95%。改对和改错以 Native 为参照。开发参照模型在正式测试之前固定。延迟为批运行秒数除以题数。",
            "", table(("任务", "指标", "n", "Native", "锁定权重", "差值和区间 pp", "改对／改错", "开发参照", "参照正式得分",
                       "Native / 锁定 ms/题"), primary),
            "", "## K2–K6 同题", "", "深度表从各预算共同题号取交集，格内是相对 Native 的差值和 95% 配对区间，单位为百分点。",
            "", table(("任务", "指标", "n", "K2", "K3", "K4", "K5", "K6"), transfer),
            "", "完整分组成绩和逐题配对区间在 results/PAIRED_CONFIRMATION.csv。源码、生成、候选分数和实际计算量随原始压缩档归档。", ""]
    Path("RESULTS.md").write_text("\n".join(lead), encoding="utf-8")


if __name__ == "__main__":
    root = Path("results")
    config = read(root / "CONFIRMATION_LOCK.json")
    dev = read(root / "PAIRED_DEVELOPMENT.json")["results"]
    final = read(root / "PAIRED_CONFIRMATION.json")["results"]
    training_tables(root, config, dev)
    evaluation_tables(root, config, final)
