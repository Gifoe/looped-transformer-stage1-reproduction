"""Render the fixed iteration-1 development gate from the audited server records."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/"iteration1"
DEPTHS=(1,2,3,4,6)
ARMS=("binding","ce","distill","dropout")


def load(p):
    return json.loads(p.read_bytes())


def sequence(xs):
    return "/".join(map(str,xs))


def main():
    g=load(OUT/"dev/COMPLETE.json")
    par=load(HERE/"NATIVE_DEPTH6_PARITY.json")
    split=load(HERE/"SPLIT_MANIFEST.json")
    r=g["screen"]
    d=g["methods"]
    names={"binding":"配对一致性","ce":"多出口 CE LoRA","distill":"K6 候选蒸馏","dropout":"Loop Dropout"}
    head="通过开发集门槛" if g["passed"] else "未通过开发集门槛"
    lines=[
        "# Ouro-1.4B 配对选项一致性适配",
        "",
        f"**{head}。** ARC-Challenge Train 内部分出 128 道开发题。候选在 K1、K2、K3、K4、K6 相对同深度最强原生或 LoopCD 的正确数差为 {sequence(r['exact_gain_by_depth'])}。相对同数据、同视图多出口 CE LoRA 的平均正确数差为 {r['mean_exact_gain_vs_CE_count']:+.2f}，两种选项顺序下选中同一内容的平均一致题数相对对应原生或 LoopCD 比较组变化 {r['mean_content_agreement_gain_count']:+.2f}。",
        "",
        "原生预算均实际执行 K×24 个物理层。训练使用一次真实 K6 轨迹的五个原生出口。四个适配臂都在全部 24 个物理层的共享 Q/V 上训练同形状的 rank-8、alpha-16 低秩矩阵。每道训练题使用原始和非恒等循环选项顺序。部署比较使用可逆合并后的 BF16 稠密权重。检查点只包含低秩更新。",
        "",
        "开发集主指标为对单个原始顺序提示无约束生成一个 token，按固定字母解析器计分。非法输出计错。限制在有效字母内的 argmax、非恒等重排和全文字符归一化均为指定诊断。评估重排独立于训练旋转。",
        "",
        "| 方法 | K1 | K2 | K3 | K4 | K6 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label,groups,tag in [("Native","base","native"),("LoopCD Fixed","base","loopcd_fixed"),
                              ("LoopCD Adaptive","base","loopcd_adaptive"),
                              ("LoopCD Hidden","base","loopcd_hidden"),
                              *((names[a],a,"native") for a in ARMS)]:
        series=[d[groups].get(f"{tag}_k{k}",{}).get("original_correct","—") for k in DEPTHS]
        lines.append("| "+label+" | "+" | ".join(map(str,series))+" |")
    lines.extend(["","下表中每个单元格按 K1/K2/K3/K4/K6 顺序列出题数。","",
        "| 方法 | 原始顺序非法输出 | 原始顺序限制字母正确 | 两顺序预测内容一致 | 两顺序均答对 | 全文字符归一化正确 |",
        "|---|---:|---:|---:|---:|---:|"])
    diagkeys=("original_invalid","original_restricted_correct","same_content","both_exact_correct",
              "content_target_char_correct")
    for label,arm in [("Native","base"),*((names[a],a) for a in ARMS)]:
        entries=[d[arm][f"native_k{k}"] for k in DEPTHS]
        cols=[sequence(m[q] for m in entries) for q in diagkeys]
        lines.append("| "+label+" | "+" | ".join(cols)+" |")
    lines.extend(["","## 同深度配对差异","",
                  "| K | 开发集最强原生/LoopCD | 新增正确 / 新增错误 | 净差百分点（配对 95% 区间） |",
                  "|---|---|---:|---:|"])
    for k in DEPTHS:
        b=g["same_budget_strength"][str(k)]
        item=g["paired_binding_vs_all"]["base"][b]["exact"]
        ci=item["paired_bootstrap95_pp"]
        lines.append(f"| {k} | {b} | {item['corrected']}/{item['damaged']} | {item['pp']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] |")
    lines.extend([
        "",
        "上表区间对应逐题配对描述。开发集按每个预算的准确率选择对应对照。正式迁移确认使用预先冻结的对照，并逐项比较各候选基线。",
        "",
        "## 训练和技术核查",
        "",
        "| 适配 | 有效训练对 / 真实提示 | 更新 | 训练秒 | 分配显存峰值 MiB |",
        "|---|---:|---:|---:|---:|",
    ])
    for a in ARMS:
        t=load(OUT/f"train.{a}.json")
        lines.append(f"| {names[a]} | {t['train_items']}/{t['prompt_views']} | {t['updates']} | {t['seconds']:.1f} | {t['peak_vram_bytes']/2**20:.0f} |")
    f=[row for rec in par["outcomes"] for row in rec["depths"].values()]
    pre=sum(x["cached_vs_uncached_prefill_top1_equal"] for x in f)
    dec=sum(x["cache_vs_full_extended_top1_equal"] for x in f)
    premax=max(x["cached_vs_uncached_prefill_max_abs_logit"] for x in f)
    decmax=max(x["cache_vs_full_extended_max_abs_logit"] for x in f)
    lines += [
        "",
        f"固定六条训练提示均验证了实际 144 层调用，手动展开与无缓存原生逐元素一致。缓存 prefill 与无缓存路径 argmax 一致 {pre}/{len(f)}，最大词表 logit 差 {premax:.4f}。强制单步 decode 与完整重算一致 {dec}/{len(f)}，最大差 {decmax:.4f}。这些是技术诊断。部署成本结论需要同样缓存路径的正式配对计时。",
        "",
        "Loop Dropout 逐提示、逐循环取独立 Bernoulli 门，全循环内各 Q/V 和位置共用同一门，存活时乘以 2。对照采用同环境的 rank-8 Q/V 和五出口 CE。原论文报告了 rank-16 七投影及终端出口训练。[Loop Dropout 原文](https://arxiv.org/html/2609.34218v1)",
        "",
        f"数据冻结含 ARC fit/dev/diagnostic 题数 {sequence(len(split['cohorts'][x]) for x in ('fit','dev','diagnostic'))}，另锁定 {len(split['cohorts']['transfer'])} 道 OpenBookQA Train。历史 ARC Train 曾整分区评分。已知历史 ARC 定向样本及重复簇不参与本次开发和诊断，OpenBookQA 迁移排除了历史测试或验证同题及近重复。",
        "",
    ]
    if not g["passed"]:
        lines += [
            "开发集门槛未通过。本轮的 ARC 256 道诊断题和 OpenBookQA 384 道迁移题保持封存。正式部署成本、跨域确认和混合循环因果诊断未启动。",
            "",
        ]
    lines += [
        "- [预先冻结的决策和门槛](ASTRA_REVIEW.md)",
        "- [输入来源与冻结划分](SPLIT_MANIFEST.json)",
        "- [真实 K6 手动展开核查](NATIVE_DEPTH6_PARITY.json)",
        "- [开发集逐题和完整配对统计](iteration1/dev/)",
        "- [执行清单](EXECUTION_FREEZE.json)",
        "",
    ]
    (HERE/"ITERATION1_REPORT_ZH.md").write_text("\n".join(lines),encoding="utf8")


if __name__=="__main__":
    main()

