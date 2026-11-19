"""Produce independent all-budget confirmation and cost gate receipts."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE=Path(__file__).resolve().parent
OUT=HERE/"iteration1"
DEPTHS=(1,2,3,4,6)
ARMS=("binding","ce","distill","dropout")


def read(name):
    return json.loads((OUT/name).read_bytes())


def seq(items):
    return "/".join(map(str,items))


def paired_exact(base,new):
    wins,losses=sum(y and not x for x,y in zip(base,new)),sum(x and not y for x,y in zip(base,new))
    n=len(base)
    counts=np.random.default_rng(101).multinomial(n,[(n-wins-losses)/n,wins/n,losses/n],size=9999)
    ci=np.quantile(100*(counts[:,1]-counts[:,2])/n,[.025,.975]).tolist()
    return {"n":n,"wins":wins,"losses":losses,"net":wins-losses,
            "p_greater":float(binomtest(wins,wins+losses,alternative="greater").pvalue) if wins+losses else 1.0,
            "paired_bootstrap_95_pp":ci}


def intra_budget(cohort):
    rows=read(f"{cohort}/predictions.binding.json")
    a=[r["original"]["native_k4"]["exact"]==r["gold"] for r in rows]
    b=[r["original"]["native_k6"]["exact"]==r["gold"] for r in rows]
    return paired_exact(a,b)


def outcome(s):
    return bool(s["net"]>0 and s["exact_one_sided_p"]<.05)


def main():
    dev,diagnostic,transfer=(read(f"{k}/COMPLETE.json") for k in ("dev","diagnostic","transfer"))
    assert dev["passed"]
    assert all(s["adapter_hashes"]==dev["adapter_hashes"] and
               s["execution_freeze_sha256"]==dev["execution_freeze_sha256"] for s in (diagnostic,transfer))
    assert transfer["input_rows"]==384 and diagnostic["input_rows"]==256
    work=read("postgate/STANDALONE_COST.json")
    cost_audit=read("postgate/STANDALONE_COST_AUDIT.json")
    comparisons=sum(cost_audit["first_token_disagreements_cached_batch1_vs_no_cache_batch_by_cell"].values())
    swap=read("postgate/RECURRENT_SWITCH.json")
    src=read("ITERATION1_AUDIT.json")
    assert work["executions"]==4736 and swap["n"]==128
    assert (OUT/"postgate/queue_complete.txt").exists()
    assert cost_audit["status"]=="PASS" and cost_audit["execution_rows"]==work["executions"]
    fp=transfer["paired_binding_vs_all"]
    family_benchmark=fp["base"]
    confirmed_native_family=all(outcome(b["exact"]) for b in family_benchmark.values())
    all_trained={arm:all(outcome(b["exact"]) for b in fp[arm].values()) for arm in ("ce","distill","dropout")}
    depth_uplift={k:intra_budget(k) for k in ("dev","diagnostic","transfer")}
    deeper=depth_uplift["transfer"]["net"]>0 and depth_uplift["transfer"]["p_greater"]<.05
    verdicts={
        "development_gate_passed":dev["passed"],
        "blind_confirm_family_all_constituent_one_sided_p_below_point05":confirmed_native_family,
        "blind_confirm_all_trained_vs_binding_families":all_trained,
        "binding_K6_better_than_own_K4_confirmed":deeper,
        "equal_work_latency_ratio_all_depths":work["all_cost_gates_pass"],
        "blind_confirm_K4_to_K6":depth_uplift["transfer"],
        "within_candidate_K4_to_K6_all_cohorts":depth_uplift,
    }
    display={"binding":"配对一致性","ce":"多出口 CE LoRA","distill":"K6 候选蒸馏","dropout":"Loop Dropout"}
    lines=[
        "# Ouro-1.4B 配对选项一致性适配的独立确认",
        "",
        "训练、开发门槛和权重已经冻结后，首次运行 384 道 OpenBookQA 官方 Train 题作为项目内盲迁移。ARC 的独立诊断划分保留 256 题。ARC Train 曾在早期项目中整体评分，故只支持本轮内部诊断。OpenBookQA 的历史验证和测试遗留 35 道重复与两道近重复 Train 题，本轮按输入相似性全部排除。未访问官方 Test。",
        "",
        "主指标是在无示例、显示候选的原始题目顺序下，无约束贪心生成一 token 后的解析正确数。部署每个预算只选择一套合并权重，使用一个提示和一次输出。下表每格按 K1/K2/K3/K4/K6 列出正确题数。",
        "",
        "| 方法 | ARC 开发 128 | ARC 内部诊断 256 | OpenBookQA 盲迁移 384 |",
        "|---|---:|---:|---:|",
    ]
    for tag,a,prefix in [("Native","base","native"), *((f"LoopCD {k.title()}","base",f"loopcd_{k}") for k in ("fixed","adaptive","hidden")),
                           *((display[a],a,"native") for a in ARMS)]:
        columns=[seq(run["methods"][a].get(f"{prefix}_k{k}",{}).get("original_correct","—") for k in DEPTHS)
                 for run in (dev,diagnostic,transfer)]
        lines.append("| "+tag+" | "+" | ".join(columns)+" |")
    lines += [
        "",
        "## 预先指定的迁移判定",
        "",
        "每个预算同时与 Native 和可运行的全部 LoopCD 变体比较。联合主张采用相交并检验，各成分要求配对单侧精确 McNemar p < 0.05 且净增题数为正。表中的区间是题级重采样描述。候选在本轮开发集上选用的最强基线保持冻结，同时报告迁移本身的观察最大值。",
        "",
        "| K | 开发冻结基线 | 迁移观察最大基线 | 候选对观察最大值纠正 / 退化 | 净差百分点 [95% 区间] | 单侧 p |",
        "|---|---|---|---:|---:|---:|",
    ]
    for k in DEPTHS:
        frozen=dev["same_budget_strength"][str(k)]
        best=transfer["same_budget_strength"][str(k)]
        pair=family_benchmark[best]["exact"]
        lo,hi=pair["paired_bootstrap95_pp"]
        lines.append(f"| {k} | {frozen} | {best} | {pair['corrected']}/{pair['damaged']} | {pair['pp']:+.2f} [{lo:+.2f},{hi:+.2f}] | {pair['exact_one_sided_p']:.6g} |")
    lines += [
        "",
        f"原生与 LoopCD 联合检验{'通过' if confirmed_native_family else '未通过'}。相同训练资源下，与普通 CE、K6 蒸馏、Loop Dropout 的五预算联合检验分别{'通过' if all_trained['ce'] else '未通过'}、{'通过' if all_trained['distill'] else '未通过'}、{'通过' if all_trained['dropout'] else '未通过'}。",
        "",
        "| 候选自身 K6 对 K4 | 纠正 / 退化 | 净变化 | 单侧 p |",
        "|---|---:|---:|---:|",
    ]
    for cohort,label in (("dev","ARC 开发"),("diagnostic","ARC 内部诊断"),("transfer","OpenBookQA 盲迁移")):
        r=depth_uplift[cohort]
        lines.append(f"| {label} | {r['wins']}/{r['losses']} | {r['net']:+d} | {r['p_greater']:.6g} |")
    lines += [
        "",
        "## 相同提示下的输出格式诊断",
        "",
        "| 方法 | 原始顺序非法输出 | 原始顺序限制字母正确 | 两套顺序同内容 | 全文字符归一化正确 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name,arm in [("Native","base"), *((display[a],a) for a in ARMS)]:
        columns=[seq(transfer["methods"][arm][f"native_k{k}"][key] for k in DEPTHS)
                 for key in ("original_invalid","original_restricted_correct","same_content","content_target_char_correct")]
        lines.append("| "+name+" | "+" | ".join(columns)+" |")
    lines += [
        "",
        "## 跨轮换权重诊断",
        "",
        "下表为 ARC 开发 128 题上的正确数。实验逐物理层切换实际合并的 Q/V 权重。相同模型 sham 在全部真实循环状态上逐元素一致，混合路径在 K1 的全部状态与指定首轮模型逐元素一致。混合路径属于机制诊断，部署路径每轮使用同一个模型。",
        "",
        "| 路径 | K1 | K4 | K6 |",
        "|---|---:|---:|---:|",
    ]
    for k, label in (("base","原生"),("binding","配对一致性"),("ce","普通 CE"),
                    ("binding_first_then_base","候选首轮，后续原生"),
                    ("base_first_then_binding","原生首轮，后续候选")):
        lines.append("| "+label+" | "+" | ".join(str(swap["correct"][k][str(j)]) for j in (1,4,6))+" |")
    lines += [
        "",
        "## 同预算实际部署成本",
        "",
        "固定 64 道开发题，每项运行两次，随机交错。以真实缓存和独立停止分别运行各个预算。等长工作包含一次 prefill 与一 token 强制解码。候选与原生共用相同的稠密形状、层和词表投影数量。权重缓存准备在计时外完成。",
        "",
        "| K | Native prefill / decode 中位 ms | 候选 prefill / decode 中位 ms | 等长工作比值中位 | 单侧 95% 上界 |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in DEPTHS:
        n=f"native_k{k}"
        base,c=(work["costs"][f"{arm}/{n}"] for arm in ("base","binding"))
        p=work["binding_vs_base_same_budget"][str(k)]
        def med(x):
            return f"{x['prefill_ms']['median']:.2f}/{x['forced_decode_ms']['median']:.2f}"
        lines.append(f"| {k} | {med(base)} | {med(c)} | {p['median_prompt_paired_ratio']:.4f} | {p['upper_one_sided_95']:.4f} |")
    lines += [
        "",
        f"五预算 1.05 延迟上界门槛{'通过' if work['all_cost_gates_pass'] else '未通过'}。全部方法的 p95、实际生成长度与分配峰值随逐次执行公开。",
        "",
        f"单题缓存计时的首 token 与批量无缓存评估有 {comparisons}/{work['executions']} 条不一致。主准确率表使用预先指定的无缓存评估。",
        "",
        "- [原始开发门槛和训练账目](ITERATION1_REPORT_ZH.md)",
        "- [OpenBookQA 按题预测与配对统计](iteration1/transfer/)",
        "- [ARC 256 题诊断](iteration1/diagnostic/)",
        "- [跨轮 shams](iteration1/postgate/RECURRENT_SWITCH.json)",
        "- [独立运行耗时](iteration1/postgate/STANDALONE_COST.json)",
        "- [代码与原始时间复核](iteration1/postgate/STANDALONE_COST_AUDIT.json)",
        "- [与普通 CE 的内容一致性和 Jensen-Shannon 散度](iteration1/BINDING_SPECIFICITY.json)",
        "- [逐题独立复核](iteration1/TRANSFER_INDEPENDENT_AUDIT.json)",
        "- [完整终局判定](iteration1/FINAL_VERDICTS.json)",
        "",
    ]
    (HERE/"FINAL_ITERATION1_ZH.md").write_text("\n".join(lines),encoding="utf8")
    (OUT/"FINAL_VERDICTS.json").write_text(json.dumps(verdicts,indent=2)+"\n",encoding="utf8")
    print(verdicts)


if __name__=="__main__":
    main()

