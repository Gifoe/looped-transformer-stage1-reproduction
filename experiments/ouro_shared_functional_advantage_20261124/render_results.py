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
        paired_ref = own["paired"][ref][metric]
        old_cost = 1e3 * cell["native"]["cost"]["seconds"] / before["n"]
        new_cost = 1e3 * own["cost"]["seconds"] / before["n"]
        primary.append((ds, METRIC_NAMES[metric], before["n"], percent(before["reference"]), percent(before["controller"]),
                        difference(before), f"{before['rescued']}/{before['harmed']}", ref,
                        percent(paired_ref["reference"]), difference(paired_ref), f"{old_cost:.1f} / {new_cost:.1f}"))
    arm_rows = []
    for arm in config["run_order"]:
        values = []
        for ds, mode, metric in ENDPOINTS:
            row = results.get(f"{ds}/k4/{mode}", {}).get(arm)
            if row is None:
                values.append("待评测")
                continue
            stats = row["metrics"]
            values.append(f"{percent(stats['acc']['value'])} / {percent(stats[metric]['value'])}" if mode == "candidate"
                          else percent(stats[metric]["value"]))
        early = results.get("gsm8k/k4/gsm_process", {}).get(arm)
        arm_rows.append((arm, *values, percent(early["metrics"]["premature_eos"]["value"]) if early else "待评测"))
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
                       "相对参照差值和区间 pp", "Native / 锁定 ms/题"), primary),
            "", "## K4 全部控制器", "",
            "候选格依次为原始 `acc` 和字符归一 `acc_norm`。内容格为完整词表贪心生成的答案匹配。标记前 EOS 越低越好。", "",
            table(("控制器", "ARC 候选", "CSQA 候选", "ARC 内容", "CSQA 内容", "SQuAD EM", "GSM8K 严格数值",
                   "GSM8K 过程联合", "GSM8K 标记前 EOS"), arm_rows),
            "", "## K2–K6 同题", "", "深度表从各预算共同题号取交集，格内是相对 Native 的差值和 95% 配对区间，单位为百分点。",
            "", table(("任务", "指标", "n", "K2", "K3", "K4", "K5", "K6"), transfer),
            "", "完整分组成绩和逐题配对区间在 results/PAIRED_CONFIRMATION.csv。源码、生成、候选分数和实际计算量随原始压缩档归档。", ""]
    Path("RESULTS.md").write_text("\n".join(lead), encoding="utf-8")


def finish_report(root, config, development, confirmation):
    selected = config["selected"]
    tables = confirmation["results"]
    coverage = confirmation["coverage"]
    if coverage["cells"] != coverage["expected"]:
        raise ValueError("Incomplete confirmation cannot populate the final research report")
    path = Path("REPORT_ZH.md")
    original = path.read_text(encoding="utf-8")
    heading, _, body = original.split("\n\n", 2)
    body = body.split("\n## 正式确认结果\n", 1)[0].rstrip()
    body = body.replace("[开发结果](DEVELOPMENT_RESULTS.md) · [训练消融](DEVELOPMENT_ABLATIONS.md)",
                        "[开发结果](results/PAIRED_DEVELOPMENT.csv) · [训练消融](ABLATIONS.md)")
    body = body.replace("正式测试若继续，将只在选型和输入冻结后运行。", "正式测试在选型、评测配置、输入和权重指纹锁定后运行。")

    ledger = Path("GPU_LEDGER.tsv").read_text().splitlines()
    elapsed = sum(int(row.split("\t")[4]) for row in ledger if row.startswith("END\t"))
    hours, minutes = elapsed // 3600, elapsed % 3600 // 60
    opening = (f"四轮研究使用单卡 RTX 4090 D 完成 GPU 验证。GPU 作业累计 {hours} 小时 {minutes} 分 {elapsed%60} 秒。"
               f"全部 {coverage['cells']} 项正式条件已通过逐题回执、冻结配置和指标重算校验。"
               f"仅用开发集选择 {selected}，冻结 Ouro-1.4B Base 并保留一个 4,096 参数检查点。")
    references = []
    for dataset, mode, metric in ENDPOINTS:
        cell = tables[f"{dataset}/k4/{mode}"][selected]
        reference = config["reference_envelope"][f"{dataset}/k4/{mode}/{metric}"]["arm"]
        native = cell["paired"]["native"][metric]
        baseline = cell["paired"][reference][metric]
        references.append((dataset, metric, native, baseline))
    native_wins = sum(native["delta"] > 0 for _, _, native, _ in references)
    baseline_wins = sum(baseline["delta"] > 0 for _, _, _, baseline in references)
    comparison = table(("任务", "指标", "n", "相对 Native pp [95% 区间]", "相对开发集选定强参照 pp [95% 区间]"),
                       [(ds, METRIC_NAMES[metric], native["n"], difference(native), difference(baseline))
                        for ds, metric, native, baseline in references])
    replay = []
    for k in range(2, 5):
        scores = development[f"gsm8k/k{k}/gsm_process"]
        old, new = (scores[name]["metrics"]["process_correct"]["count"] for name in ("mixed", "anchored_policy"))
        replay.append(f"K{k} {old}→{new}/64")
    depth_panel = read(root / "MATCHED_DEPTH.json")
    contrast_rows = []
    for label, key, metric in (("ARC 候选", "arc_challenge/candidate", "acc_norm"),
                               ("ARC 自由生成", "arc_challenge/content", "correct"),
                               ("GSM8K 过程联合", "gsm8k/gsm_process", "process_correct"),
                               ("GSM8K 标记前 EOS", "gsm8k/gsm_process", "premature_eos")):
        filtered = depth_panel[key]
        contrast_rows.append((label, len(filtered['ids']),
                              *(difference(filtered['depths'][f'k{k}']['native'][metric]) for k in (2,3,4,5,6))))
    depth_table = table(("同题端点", "n", "K2", "K3", "K4", "K5", "K6"), contrast_rows)
    sections = ["## 正式确认结果", "",
                f"第四轮开发集补齐后，普通混合到第四轮的 GSM8K 过程联合正确分别为{'、'.join(replay)}。"
                f"开发集最弱端点相对基线包络的差值为 {100*config['validation'][selected]['worst_delta']:+.2f} 个百分点。", "",
                f"在 K4 的七个预先指定端点中，锁定权重比 Native 高 {native_wins}/7 项，比各端点的开发集强参照高 {baseline_wins}/7 项。"
                "这些计数是点估计，逐项区间见下表。", "", comparison, "",
                "## 跨深度的候选与真实生成", "",
                "各深度使用相同题号。格内为锁定权重相对 Native 的百分点及 95% 配对区间。标记前 EOS 的正差为退化。",
                "", depth_table, "",
                "ARC-Challenge Test 排除一条已记录的近重复题。CSQA 使用公开的作者 Validation 作为固定评测集，相关题集曾出现在本项目系列中。"
                "SQuAD 为固定 360 题，GSM8K 的 K4 过程评测为 96 题。其余深度从正式池固定抽取同题子集。"
                "泄漏检查覆盖可见的控制器训练输入，基座预训练曝光不可核实。正式延迟在恢复后的同一实例上测量。", "",
                "[全方法成绩、真实成本和同题深度分析](RESULTS.md) · [数据重叠审计](data/CONTAMINATION.json)", ""]
    path.write_text(f"{heading}\n\n{opening}\n\n{body}\n\n" + "\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    root = Path("results")
    config = read(root / "CONFIRMATION_LOCK.json")
    dev = read(root / "PAIRED_DEVELOPMENT.json")["results"]
    final = read(root / "PAIRED_CONFIRMATION.json")
    training_tables(root, config, dev)
    evaluation_tables(root, config, final["results"])
    finish_report(root, config, dev, final)
