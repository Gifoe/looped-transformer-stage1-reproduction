"""Render claims only from complete measured outputs."""
import argparse
from pathlib import Path

from native import load

FAMILIES=(("openbookqa","OpenBookQA"),("arc_challenge","ARC-Challenge"))


def val(r,d=4):
    if r is None:
        return "—"
    return f"{r['mean']:+.{d}f} [{r['ci'][0]:+.{d}f}, {r['ci'][1]:+.{d}f}]"


def table(header,lines):
    return "\n".join(["","| "+" | ".join(header)+" |","| "+" | ".join(["---"]*len(header))+" |",
        *["| "+" | ".join(map(str,line))+" |" for line in lines],""])


def build(o):
    s=load(o/"SUMMARY.json")
    m=load(o/"MECHANISM_SUMMARY.json")
    r=s["structure"]["confirmation"]
    text=["# Ouro-1.4B 固定路由和受体内容",
          "",
          "基于冻结 Ouro-1.4B Base 的 574fa66 checkpoint 和已公开的上一轮实验。物理 L6 的四头 1、7、11、14（零基）来自上一轮未使用标签的筛选。"
          "本轮落实其最强语义判别，并在新的自然问答上追溯第一层实测写入。",
          "",
          "## 结论",
          ""]
    pass_gate=s["structured_decision"]["preregistered_gate"]
    primary=r["all"]["panels"]["native_values"]
    baseline=r["all"]["panels"]["native_values_improvement_over_scramble_values"]
    text.extend([
        ("固定路由与受体 V 的规则交互通过预登记门槛。" if pass_gate else "冻结的 L6 注意力路由没有通过规则交互门槛。")+
        f"固定前缀、交叉 A 与 V 的特异交互为 {val(primary['interaction'])} nats/token，相对于同剂量按行交换对照为 {val(baseline)}。",
        "在自然提示中，单个早期写入会改变较晚轮次的注意力分布和答案证据。下述删写入、复位和同剂量方向结果确定其因果范围；"
        "全候选匹配的判断与自由生成能力分开，单独的路由形成不等同于答对机制。",
        "",
        "## 1. 同一个注意力图能否使用两种受体内容",
        "",
        "供体答案与四个预测目标完全不同，查询变化只有一个对齐 token。完整受体条件交换 A；固定前缀条件只在同一接收状态中交叉四头的 A 和 V。"
        "六个候选均为两 token 的完整答案，以未加选项的提示计平均 log likelihood。配对 I 使用两个正确受体对比减去两个交叉受体对比。",
    ])
    rows=[]
    for layout,z in r.items():
        for key in ("natural_receivers","native_values","scramble_values","other_heads_receivers","native_selector_oracle"):
            p=z["panels"][key]
            rows.append((layout,key,z["n"],val(p["interaction"]),*(val(x) for x in p["matched"]),
                         p["target_hits"],p["donor_hits"],p["strict_all_correct"]))
    text.append(table(("布局","方法","组数","特异交互 I","正确受体对比 0","正确受体对比 1","目标首选/4n","供体首选/4n","四角全中/组"),rows))
    text.extend([
        "同剂量随机对照逐行交换原先两个真实 softmax 分布，保持有效因果支持、行和为 1、两图差的 Frobenius 范数不变。其他头仅在无需外推时精确匹配剂量。"
        "改变整段提示查询的原生参照同时允许 Q/K/V、残差及剩余算子共同适应。",
        "",
        "## 2. 同题真实纠正的回拨与复位",
        "",
        "两数据集按 ID、题干和旧暴露表在接触确认分数前选题。OpenBookQA 验证排除了已知 432 条历史记录和上一轮输入，ARC 合并留出仅保证本轮队列不重叠，"
        "存在更早项目整划分评测。旧排除表修正前的试运行不进入分析。模型执行不读取 gold。平均完整候选 LL 是主协议，候选总 LL 在 JSON 中独立保留。",
    ])
    rows=[]
    for fam,label in FAMILIES:
        n=s["natural"][fam]["mean"]
        x=n["panels"]["WC"]["transitions"]
        p=n["populations"]
        rows.append((label,n["n"],*n["correct"],*(p[k] for k in ("WC","CW","CC","WW")),x["rollback_damages"],
                     x["exclusive_rollback_damage"],x["dropped_late_correct"],x["restore_after_drop"],
                     x["restore_after_drop_control_specific"]))
    text.append(table(("数据","n","K1 正确","K2 正确","错→对","对→错","对→对","错→错","错→对中回拨损伤","且对照均答对",
          "错→对中删写入损伤","清洁 A 复位救回","且其他头未救回"),rows))
    rows=[]
    for fam,label in FAMILIES:
        n=s["natural"][fam]["mean"]
        for pop in ("ALL","WC","CC"):
            d=n["panels"][pop]
            for k in ("rollback","position0","rest","other_heads","drop","matched_writer","half_drop","restore","restore_other","early_add"):
                z=d[k]
                rows.append((label,pop,k,z["n"],val(z["margin_vs_receiving_native"]),z["damage"],z["rescue"],z["ood2x"]))
    text.append(table(("数据","原生状态","操作","n","对原生 margin","损伤","救回","状态范数 >2×"),rows))
    rows=[]
    for fam,label in FAMILIES:
        d=s["natural"][fam]["mean"]
        for pop in ("ALL","WC"):
            for k in ("restore_vs_drop","restore_vs_restore_other","restore_pos0_vs_drop","restore_rest_vs_drop","position0_vs_rest"):
                rows.append((label,pop,k,val(d["panels"][pop]["contrasts"][k])))
    text.append(table(("数据","状态","配对校验","margin 变化"),rows))
    text.extend(["完整回拨使用真实 K1 的 A，位置 0 条件保留其余键的相对比例，非位置 0 条件保留位置 0 的概率。"
                 "位次 0 是提示第一个实际 token，未添加 BOS。这里的复位指以删写入的接受状态和真实剩余计算为背景，单独换回清洁的 K2 注意力图。",
                 "", "## 3. 自然 A/V、Gate/Up 和第一层删除背景的四角",
                 "",
                 "同题所有四角使用相同 K2 接收前缀。交互为 joint−first−second+base。正确答案的对手在看干预结果之前固定为原生 K2 分数最高的错选项。",
                 "L6 两种头组保留各自真实因子，L1 MLP 替换真实预激活 gate 和 up。生产者删除四角的基线是删写入后的 K2，另换回清洁 A、V 或两者。"])
    rows=[]
    for fam,label in FAMILIES:
        for locus,p in m[fam]["factorials"].items():
            for name in ("ALL","WC"):
                q=p[name]; x=q["arms"]
                rows.append((label,locus,name,q["n"],val(x["first"]["effect"]),val(x["second"]["effect"]),
                             val(x["both"]["effect"]),val(q["interaction"]),"/".join(str(x[j]["hits"]) for j in ("base","first","second","both")),
                             q["only_joint_rescue"]))
    text.append(table(("数据","位置","状态","n","第一因子","第二因子","两者","交互","四角正确数","只有联合才救回"),rows))
    text.extend(["## 4. 哪些早期 token 写入形成下一轮位置 0 路由",
                 "",
                 "直接删除 K1 L1 真实写入的指定 token 支持，复跑后续各层。每类支持保留同范数坐标置换对照。"
                 "提示首 token 等义改写仅保留 BPE 严格一位对齐，通过校验的分母单独给出。"])
    rows=[]
    for fam,label in FAMILIES:
        for mode,d in m[fam]["writer"].items():
            if not d["n"]:
                rows.append((label,mode,0,"—","—","—","—","—"))
                continue
            for arm,q in d["panels"]["ALL"]["arms"].items():
                rows.append((label,mode,d["n"],arm,val(q["mass"]),val(q["factor_loss"]),val(q["margin"]),q["damage"]))
    text.append(table(("数据","提示","n","写入支持","K2 位置0 质量下降","自然方向投影损失","任务 margin","损伤"),rows))
    match=o/"MARGIN_MATCHING.json"
    if match.exists():
        a=load(match)
        text.extend(["## 5. 纠正特异性与答案边界",
             "作为事后敏感性核验，为每道错→对题在对→对题中匹配同一 clean K2 margin 和相近提示长度。匹配无放回并保留未匹配计数。"
             "比较写入删除相对于同范数坐标置换的 margin 降低，在两类题上是否不同。"])
        text.append(table(("数据","配对数","可用错→对","错→对的方向特异损失","对→对","两者差"),
          [(label,a[f]["n"],a[f]["available_wc"],val(a[f]["summary"]["drop"]["wc"]),val(a[f]["summary"]["drop"]["cc"]),
            val(a[f]["summary"]["drop"]["wc_minus_cc"])) for f,label in FAMILIES]))
    b=load(o/"BILINEAR_SUMMARY.json")
    text.extend(["## 6. 真正的乘积项与后续非线性",
        "直接在原生 o_proj 或 down 输入处加入精确乘积交互，并与原始输出加和、等范数坐标置换比较。状态和其余算子保持同一个真实 K2 接收方。"
        "总体四角效应等于真实乘积项与剩余非线性之和。剩余含后续网络、归一化和评分。"
        "由于这组分析是在见到自然联合效应之后设计，它属于已见队列的机制定位。"])
    rows=[]
    for family,label in FAMILIES:
        for site,x in b[family].items():
            for pop in ("ALL","WC"):
                q=x["groups"][pop]
                rows.append((label,site,pop,q["n"],*(val(q[k]) for k in ("total","product","downstream","product_vs_same_norm")),
                    q["hits"]["hit_both"],q["hits"]["hit_add"]))
    text.append(table(("数据","位置","状态","n","完整交互","原生乘积项","后续非线性","乘积相对同剂量对照","完整联合正确","仅加和正确"),rows))
    if (o/"SELECTION_AUDIT.json").exists():
        a=load(o/"SELECTION_AUDIT.json")
        text.extend([f"首遍清单有 {a['preliminary_obqa_overlap']} 个 OpenBookQA ID 未满足历史排除规则。该遍保留在服务器作校验，主发布仅含基于完整排除表重算后的队列。",
        "","## 核验与解释边界",""])
    text.extend(["逐题原始分数、布局、预测和锁均在 results。所有区间为题或世界组配对自举。",
                 "本轮评测自然原生纠正采用完整候选的 teacher-forced 诊断，不能替换对开放自由生成的实证。"
                 "位置/成分干预均是特定剂量和支持下的反事实。决策层边界、证据值的积累和多因子共同变化仍可能同时影响结果。"
                 "四头的结论只用于这一条具体机制，不能直接延伸到其他头或模型。",
                 "",
                 "[上一次原生状态研究](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/9ed73cd0e6568f1b23b4bdb846732b7257def107/experiments/ouro_native_state_computation_20261121)"])
    return "\n".join(text)+"\n"


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("report",type=Path)
    a=p.parse_args()
    a.report.write_text(build(a.out),encoding="utf-8")

