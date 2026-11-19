"""Write the mechanism report from the released numeric estimates."""
import argparse
import json
from pathlib import Path

from mechanism_design import load


def ci(v,i=-1,j=None):
    def pick(k):
        x=v[k] if j is None else v[k][j]
        return x[i] if isinstance(x,list) else x
    return f"{pick('mean'):+.3f} [{pick('lo'):+.3f}, {pick('hi'):+.3f}]"


def pct(v):
    return f"{100*v:.1f}%"


def acc(v):
    return " → ".join(pct(x) for x in v["mean"])


def fmt_tbl(columns, rows):
    result=["| "+" | ".join(columns)+" |","|"+"|".join("---" for _ in columns)+"|"]
    result.extend("| "+" | ".join(map(str,row))+" |" for row in rows)
    return "\n".join(result)


def judged(x,i=-1):
    return "区间在零以上" if x["lo"][i]>0 else ("区间在零以下" if x["hi"][i]<0 else "区间跨零")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("summary",type=Path)
    parser.add_argument("report",type=Path)
    x=parser.parse_args()
    r=json.loads(x.summary.read_text(encoding="utf-8"))
    tables=r["interchanges"]
    routes=r["routes"]
    open_k1=tables["new_structure/open/1/tableau_hit"]["mean"][-1]
    open_native=tables["new_structure/open/1/natural_hit"]["mean"][-1]
    open_k2=tables["new_structure/open/2/tableau_vs_random"]
    k1_control=tables["new_structure/open/1/tableau_vs_same_y_mapping_control"]
    sem_alpha=tables["new_structure/open/1/tableau_vs_base"]
    matched_h=r["same_horizon"]
    key="new_structure/open/"
    m12=matched_h[key+"one_more_loop_after_K1"]
    m23=matched_h[key+"one_more_loop_after_K2"]
    m34=matched_h[key+"one_more_loop_after_K3"]
    gain12=matched_h[key+"K2_minus_K1_with_one_more_loop"]
    gain34=matched_h[key+"K2_minus_K3_with_one_more_loop"]
    valid_composition=all(tables[key+"1/"+k]["lo"][-1]>0 for k in
        ("tableau_vs_base","tableau_vs_random","tableau_vs_same_y_mapping_control",
         "tableau_vs_source_only","tableau_vs_rule_only"))
    valid_depth=bool(gain12["lo"][0]>0 and gain34["lo"][0]>0)
    late_route=all(routes[key+k]["lo"][-1]>0 for k in
        ("source_path_specific_damage","restore_same_receiver"))
    native_route=routes[key+"readout_recut_reduces_restoration"]
    post_cut=routes[key+"readout_recut_without_prior_cut"]
    stable_recut=bool(native_route["lo"][-1]>0 and post_cut["lo"][-1]>0)
    new_p=r["paired_protocol_routes"]["new_structure"]
    text=[]
    add=text.append
    add("# Ouro-1.4B 的跨循环组合与因果路由")
    add(f"""在 48 组独立增跳题上，将分别只改变来源或规则的供体状态写入 K1 后的相同角色，K4 对两个预定新组合答案的候选准确率为 **{open_k1*96:.0f}/96（{pct(open_k1)}）**。同样问题的原生新组合世界是 **{open_native*96:.0f}/96（{pct(open_native)}）**。联合写入相对干净接收方的新结果优势为 **{ci(sem_alpha)} nats**，相对等幅同 Y 供体为 **{ci(k1_control)} nats**。{'新组合效应高于全部指定对照。' if valid_composition else '严格组合判据至少有一项对照结果未达到。'} 与结果无关的标签置换保持共享题干状态逐位一致。HotpotQA 证据替换验证了具体证据的行为作用，受控任务的 K2 路径未达到跨任务复现门槛。""")
    add("## 1. 冻结、原生计算与留出")
    add("""固定 ByteDance/Ouro-1.4B Base 574fa66cb8bf5abdc979642d01cf2b79b16bfab1，BF16、原生 SDPA、无 KV cache。24 个物理层共享于四轮。单题原生包装器分别在 K1–K4 逐位复现读出。20 路同宽批次逐题验证自替换和整边界供体替换。循环边界干预发生在末层归一化之后，未改权重和 token 序列。自然表使用九路 batch，捐献与接收采用同宽 20 路 batch；不同 batch 形状在分数并列附近可产生不同答案排序。完整替换的逐位比较在 20 路 batch 内完成。

每个合成题组有 3 个独立来源值、3 个独立规则排列，完整枚举 9 个世界。来源 token 位于规则区之前。基准答案为第 0 个语义值，两个同索引非零联合条件的正确答案与基准、两项单因素答案均不同。开发 24 组，新实体 48 组，新结构 48 组。新结构同时改变人物与符号域、加入桥接规则。新实体与新结构间共享候选词 rabbit，按题组独立，并使用互不重叠的人名与键域。三种原生机制协议分别为开放内容、有列表的内容、同列表的字母，和 ARC 的六协议受众及 few-shot 设置分开报告。每个估计按题组重抽样 3,000 次。候选首选并列记作未答对。""")
    bs=[]
    for split in ("development","new_entities","new_structure"):
        for p in ("open","content","letter"):
            z=r["behavior"][f"{split}/{p}"]
            bs.append((split,p,z["answers"],acc(z["accuracy"]),acc(z["greedy_accuracy"])))
    add(fmt_tbl(("分割","协议","世界","候选正确率 K1–K4","无约束首 token 命中"),bs))
    add("[运行和假设冻结](MECHANISM_FREEZE.md) · [逐世界自然结果](results/mechanism/natural.jsonl)")

    add("## 2. 原生状态中可以观测到哪些变化")
    add("""完整 token 留档覆盖两个同题自然世界与全部 96 层事件，包含残差、Q/K/V 投影、attention/MLP 输出和实际加入残差的归一化更新。360 个题组×协议表持续追踪各角色的未投影宽度响应，27 个题组×协议另存逐 token 的源、规则、同 Y 与交互差分。18 个采样题组×协议另存 K2 切断、恢复以及 K3 再切断造成的逐 token 差分。同一物理层不同轮次的输入差分与两条实际更新差分可在数组逐项核对。差分 RMS 有助于定位传播。对某一正确答案的证据则来自固定语言模型 head 的有符号分数与原生答案。

原生早轮候选得分以完整 token 概率计算，中间层使用固定原生终端归一化和输出 head。早层分数可用于外部观测。后期出现可读的结果，无法凭读出值认定首次形成时点。相同正确答案的供体能产生非零状态距离，交互范数也可能非零。两者不能判定模型已组合出指定答案。""")
    add("[图 2 角色与物理层](figures/mechanism/figure_2_token_role_atlas.pdf) · [图 2b 逐 token 因果轨迹](figures/mechanism/figure_2b_full_token_atlas_example.pdf) · [可执行因果图谱](CAUSAL_COMPUTATION_ATLAS/README.md)")

    add("## 3. 共享循环体能在别处续算新结果")
    add("""对每个边界，用世界 (s,0) 的来源 token 状态结合世界 (0,m) 的完整规则块。正确的联合目标世界 (s,m) 从未用于对应两块供体的选取。在增跳题中保留接收方的桥接区和读出位置。控制包括自替换、独立单因素、等幅随机、同 Y 规则供体、来自联合世界的匹配规则块，以及完整边界恢复。完整边界条件直接复现原生供体，是测量上限，对局部可分性不构成支持。""")
    ts=[]
    for split in ("new_entities","new_structure"):
        for p in ("open","content"):
            for loop in (1,2,3):
                z=f"{split}/{p}/{loop}/"
                ts.append((split,p,f"K{loop}",pct(tables[z+"tableau_hit"]["mean"][-1]),
                           ci(tables[z+"tableau_vs_random"]),ci(tables[z+"tableau_vs_same_y_mapping_control"]),
                           ci(tables[z+"tableau_vs_source_only"]),ci(tables[z+"tableau_vs_rule_only"])))
    add(fmt_tbl(("分割","协议","写入","K4 指定答案正确率","vs 等幅随机","vs 同 Y","vs 仅来源","vs 仅规则"),ts))
    add(f"""将后续计算统一为一次循环后，结构留出开放协议在 K1→K2、K2→K3、K3→K4 的指定答案命中依次为 **{pct(m12['mean'][4])}、{pct(m23['mean'][4])}、{pct(m34['mean'][4])}**。相比等幅随机方向，三个转换的目标 margin 效应为 **{ci(m12,0)}、{ci(m23,0)}、{ci(m34,0)} nats**。K2→K3 与两端相同时长转换的配对差依次为 **{ci(gain12,0)}、{ci(gain34,0)} nats**。{'两项配对差均支持阶段选择性。' if valid_depth else '匹配剩余时间后，K2 峰值的两项独立配对门槛未同时满足。'} K2 边界改动对 K4 的新组合效应相对随机为 {ci(open_k2)} nats。""")
    add("""这个实验能反驳纯粹复制供体最终决策的过程。每块局部状态可以携带多个可能结果，以及用于筛选它们的上下文。转换所需信息包括指定来源 token、规则块和保留的其他状态。共同共享的 Transformer 权重在下一轮执行这批输入时会改变真实答案。具体路线可以跨物理层以及未干预的状态并行传播。实验没有将全网还原为唯一算子序列，也无法判定自然世界中是否同时携带早期形成的结果。""")
    gid="new_structure-000"
    rdir=x.summary.parent
    example=next(g for g in load(rdir/"mechanism_tableaux.jsonl") if g["id"]==gid)
    native={(a["s"],a["m"]):a for a in load(rdir/"natural.jsonl") if a["id"]==gid and a["fork"]=="open"}
    patched={(a["s"],a["m"]):a for a in load(rdir/"causal.jsonl") if a["id"]==gid and a["fork"]=="open"
             and a["loop"]==1 and a["arm"] in ("tableau","self")}
    names=example["answers"]
    case_rows=[]
    for i in range(3):
        for j in range(3):
            n,b=native[i,j],patched[i,j]
            target=names[n["y"]]
            value=lambda z: names[z["pred"][-1]] if z["pred"][-1]>=0 else "并列"
            case_rows.append((example["keys"][i],j,target,value(n),value(b)))
    add(f"预先按 ID 取增跳留出的第一组 {gid}。{example['name']} 的来源键经桥接符号进入物品规则；表中展示该组完整九个世界的 K4 候选首选。")
    add(fmt_tbl(("输入键","规则索引","预定答案","原生 K4","K1 分开供体写入后的 K4"),case_rows))
    add("[图 6 组合与对照](figures/mechanism/figure_6_competing_mechanisms.pdf) · [图 6b 同等剩余深度](figures/mechanism/figure_6b_matched_remaining_depth.pdf) · [逐条置换和路由结果](results/mechanism/causal.jsonl)")

    add("## 4. K2 信息的延迟使用与 K3 再切断")
    add("""K2 阻断具体来源键到规则区域的注意力，对照阻断同数目的无关来源键。把同一接收方干净的规则区恢复到 K2 边界。K3 再切断规则区到查询和终端读出，并对照相同数量的其他前缀键。再切断的恢复差分分别在无恢复和有恢复的同一类掩码中计算，因此后续路线的基线主效应不会被计入恢复损失。""")
    cs=[]
    combos=("source_path_specific_damage","restore_same_receiver","readout_recut_reduces_restoration",
            "readout_recut_without_prior_cut")
    labels=("来源切断相对无关键","同接收方恢复","K3 规则再切断减少恢复","无前置切断的 K3 规则再切断")
    for p in ("open","content"):
        for q,label in zip(combos,labels):
            z=routes[f"new_structure/{p}/{q}"]
            cs.append((p,label,ci(z,1),ci(z,2),ci(z,3)))
    add(fmt_tbl(("协议","配对边效应（nats）","K2","K3","K4"),cs))
    add(f"""开放结构留出的 K2 源路径在 K4 的损害和局部恢复{'均通过指定方向的区间门槛' if late_route else '未同时通过指定方向的区间门槛'}。再切断恢复差在 K4 {judged(native_route)}，无前置干预的直接再切断差 {judged(post_cut)}。{'开放路径的 K4 完整串联判据通过。' if stable_recut else '完整串联到最终答案的假设仍有未通过的因果控制。'} 内容列表相较开放协议对 K4 来源切断损害的配对差为 {ci(new_p,3,0)} nats，恢复差为 {ci(new_p,3,1)} nats。该比较区分可见候选对这条具体路径的影响和跨协议总体答题分数变化。""")
    add("[图 4 延迟使用与真实任务](figures/mechanism/figure_4_delayed_causal_use.pdf)")

    add("## 5. 功能对齐与反事实分叉")
    add("""对每组题在相同语义候选坐标中计算 9×5 有符号答复响应。剔除每个世界的整体偏置及固定候选偏置。跨协议使用角色匹配的最终读出和同一语义目标，匹配依据是开发 split 上的响应指纹。新实体与增跳结构检验匹配是否优于同一物理层。未来行为预测只取当前预算以内的开放协议事件，以本轮原生答案 margin 为基线，并分别评估纠正与保留。""")
    ph=[]
    for p in ("content","letter"):
        for split in ("new_entities","new_structure"):
            d=r["phase"][p]["held_out"][split]
            fc=[z["correction"] for z in d["forecast"]]
            ph.append((p,split,ci(d["mean_match_gain"]),
                " / ".join("—" if z["aligned_open_margin_auc"] is None else f"{z['aligned_open_margin_auc']:.3f}" for z in fc),
                " / ".join("—" if z["current_native_margin_auc"] is None else f"{z['current_native_margin_auc']:.3f}" for z in fc)))
    add(fmt_tbl(("与开放协议配对","留出","匹配减同层","纠正 AUC K1→2 / K2→3 / K3→4","本轮 margin AUC"),ph))
    q=r["forks"]["new_structure/relabel/gold_new_letter_minus_original"] if "new_structure/relabel/gold_new_letter_minus_original" in r["forks"] else r["forks"]["new_structure/relabel/correct_label_vs_old_label"]
    label_new=r["forks"]["new_structure/relabel/new_label_selected"]
    label_old=r["forks"]["new_structure/relabel/old_label_selected"]
    order=r["forks"]["new_structure/order/content_choice_flip"]
    relabel=r["forks"]["new_structure/relabel/content_choice_flip"]
    add(f"""每题七个自然后缀保持题干 token 逐位相同，所有 {r['counts']['atlas_sites_per_shared_prefix']} 个原生事件的共享前缀状态均逐位相同。自然分支比较无列表、固定列表、同列表字母目标、替换干扰项、保持绑定重排、保持顺序改标签，以及新增无关证据。增跳组 K4 的改序／改标签内容首选变化分别为 {pct(order['mean'][-1])}、{pct(relabel['mean'][-1])}。在正确标签确实改变的 {q['n']} 组中，K4 选择新标签比例为 {pct(label_new['mean'][-1])}，选择旧标签比例为 {pct(label_old['mean'][-1])}，新标签相对旧标签的分数差为 {ci(q)} nats。改标签分支按内容目标提示，字母分数用于诊断绑定。候选后缀改变读出和角色匹配的下游状态，共享题干持续使用相同因果输入。""")
    add("[图 3 功能响应](figures/mechanism/figure_3_functional_alignment.pdf) · [图 5 自然分叉](figures/mechanism/figure_5_counterfactual_forks.pdf) · [图 5b 后缀与字母绑定的逐层轨迹](figures/mechanism/figure_5b_suffix_and_binding.pdf) · [分叉精确检验](results/mechanism/prefix_parity.json)")

    add("## 6. 真实多跳任务的迁移边界")
    add("""官方 HotpotQA distractor validation 中，第一批选 64 道有明确跨文档支持和字面桥接的题。第二批另外选 48 道，并要求两个错误选项出现在可见无关摘录中，桥接标题未被问题直接给出。两批保留原题和支持句，构建有自然来源依据的证据角色和标量候选评价。第一批的错误选项不在可见证据，导致 K4 候选判别接近饱和。上下文为两条支持文档的标注句与无关文档摘录，替换按同 token 数执行。这里报告这两批选择性子集，不将准确率当作标准 HotpotQA 官方全量指标。""")
    ht=[]
    for nm,s in (("64 条初始集",r["hotpot"]),("48 条难对照集",r.get("hard_hotpot"))):
        if not s:
            continue
        for p in ("open","content","letter"):
            ht.append((nm,p,acc(s[f"{p}/native_accuracy"]),ci(s[f"{p}/source_replacement_damage"]),
                       ci(s[f"{p}/mapping_replacement_damage"]),ci(s[f"{p}/route_cut_damage"]),ci(s[f"{p}/route_restore"])))
    add(fmt_tbl(("队列","协议","候选正确率 K1–K4","K4 桥接替换损害","K4 答案证据替换损害","K4 特定路径切断","K4 同接收方恢复"),ht))
    add("""真实桥接证据作用可以跨任务和特征族出现，但替换整份证据可能改变直接解答线索和语言分布。固定 K2 路径的强跨任务复现要求切断相对无关键的损害和同接收方恢复都朝指定方向，并有行为支持。真实任务未达到这一条件，因此证据支持内容依赖，不支持固定物理程序在真实任务上通用。""")
    if "free_generation" in r:
        g=[]
        for split in ("new_entities","new_structure","hotpot_bridge_validation_hashed"):
            for p in ("open","content","letter"):
                z=[r["free_generation"][f"{split}/{p}/K{i}"] for i in (1,2,3,4)]
                g.append((split,p," → ".join(f"{v['correct']}/{v['n']}" for v in z)))
        add("""另对按哈希选中的 12 道受控组／16 道真实题在各预算实际解码 8 token。内容取第一项完整匹配词，字母取短回答中的首标签。该诊断保留原始续写与解析，便于检查解析界限和短生成漏判。""")
        add(fmt_tbl(("分割","协议","8-token 自由回答 K1–K4"),g))
        add("[自由解码记录](results/mechanism/free_generation.jsonl)")

    add("## 7. 计算模型的有效范围")
    add("""我们识别的是一个可检验的局部组合规律。来源与规则状态可以独立作用于原生后续循环，联合操纵能产生预先指定、未被任一供体单独指定的结果。它使用的是完整角色 token 状态，在 3×3 世界和结构留出上逐条验证。早轮读出、干预后的组合准确率和局部路由恢复分别约束可观察信息、计算的可操控性和状态的因果用处。它们没有共同确定一个手写等价的全网络程序。""")
    add("""原生输出的跨协议内部状态变化有证据，固定阶段提前需要留出阶段匹配及下一轮行为预测同时支持。功能对应没有达到该双门槛时，应保留多个在后缀内发生的路径，包括候选条件化、内容和字母绑定，以及同一信息从不同位置进入读出。固定 K2 路径的真实多跳迁移和可直接替代循环的性能提升保持为各自要验证的命题。""")
    add("""交叉供体移植属于已有因果抽象方法。前序 Ouro 四世界实验中，主要联合改动回到同一答案，正的非加性交互仍可能来自两项单独损害。本次选择联合目标和任一单因素目标都不同的两个 3×3 对角条件，直接验证预定的新答案。Wu 等已在任务微调的 Ouro-2.6B 和拟合状态控制器上，用注意力模式移植区分重新路由和复制答案。Mu 和 Wang 已在预训练 Ouro-1.4B 上用跨轮末位 query 移植改变合成两跳检索。本次干预的是同一循环边界的完整角色状态，来源与规则分别取自自然供体。匹配剩余深度的行为比较约束的是这两个角色能否在原生共享循环中继续组合。 [因果抽象](https://arxiv.org/abs/2106.02997) · [Shared Weights, Appendix D](https://arxiv.org/html/2609.39892v1) · [Cross-Loop Query Handoff](https://arxiv.org/abs/2610.34639)""")
    add("[完整可复核统计](results/mechanism/mechanism_summary.json) · [数据与运行索引](CAUSAL_COMPUTATION_ATLAS/ATLAS_INDEX.csv) · [复现脚本](reproduce_mechanism.sh)")
    x.report.write_text("\n\n".join(text)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
