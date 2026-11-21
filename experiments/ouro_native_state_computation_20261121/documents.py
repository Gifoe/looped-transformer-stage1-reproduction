"""Generate the Chinese main report and its evidence appendices from the completed run."""
import argparse
from pathlib import Path

from native import load


FAMILIES=(("confirm_arc","ARC-Challenge"),("confirm_external","OpenBookQA"),("controlled","生成结构"))
MODES=("attention","mlp")
PROT=(("gen","自由生成"),("tf","完整候选平均 LL"),("tf_sum","完整候选总 LL"),("letter","字母辅助"))


def interval(x,places=3):
    if x is None:
        return "—"
    return f"{x['mean']:+.{places}f} [{x['ci'][0]:+.{places}f}, {x['ci'][1]:+.{places}f}]"


def table(headers,rows):
    return "\n".join(["","| "+" | ".join(headers)+" |",
        "| "+" | ".join(["---"]*len(headers))+" |",
        *["| "+" | ".join(map(str,row))+" |" for row in rows],""])


def doc(root,name,title,parts):
    (root/name).write_text("# "+title+"\n\n"+"\n\n".join(parts)+"\n",encoding="utf-8")


def behavior_table(phases,flatten=False):
    rows=[]
    for p,f in FAMILIES:
        b=phases[p]["behavior"]
        for key,caption in PROT:
            d=b[key]
            rows.append([f,caption,d["n"],*d["correct"]])
    return table(["样本","协议","n","K1","K2","K3","K4"],rows)


def reports(o,root):
    s=load(o/"SUMMARY.json")
    phases=s["phases"]
    c=s["candidates"]
    ar=c["candidate_attention_layer"]; ml=c["candidate_mlp_layer"]
    fs=s["formation"]
    lock=s["origin_lock"]
    al=lock["mode"]["attention"]["selected"]
    mw=lock["mode"]["mlp"]["selected"]
    nat_para=("模型 ByteDance/Ouro-1.4B Base 固定到 `574fa66`，原生 no-cache eager，24 个物理层各两次残差写入。"
        "四轮重复同一组权重，只有末端 RMSNorm 输出传入下一轮。每轮 early-exit gate 只计算辅助读出。"
        "三组技术输入在 FP32 与 BF16 共六次检验，带记录器和不带的所有输出与固定位置完整词表 logits 逐位一致。"
        "同一输入状态、同一 RoPE 和 mask 下，强制在四个不同原生轮次使用同一状态，完整输出也逐位一致。")
    att_text=(f"盲选全 token 比例在物理 L{ar} 达最大，四个 0-based heads 为 `{c['heads_zero_based']}`。"
        "在 FP32 保存 RoPE 后 Q/K、V、softmax 后 A、A@V、实际输出投影和写入前后需用状态。"
        "先精确分解 Δ(AV) 为 ΔA·V₁、A₁·ΔV、ΔA·ΔV。输出投影是线性的，三项通过同一投影后仍相加。"
        "post-RMSNorm 另以单项替换和剩余量核算，剩余来自归一化非线性。L2 比例允许超出 1，项之间会抵消。")
    mlp_text=(f"同样准则冻结物理 L{ml} 的完整分布式 gate。投影前差为 ΔG⊙U₁ + G₁⊙ΔU + ΔG⊙ΔU，"
        "G=SiLU(gate_proj(Norm(h)))，Up 读取相同输入。down 投影与实际 post-attention 第二 norm 都保留。"
        "SiLU gate 是带符号连续乘子，符号变化也已记录。整层 gate 的结果不能分配给某个单独神经元。")
    tables={}
    for mode,title in (("attention","Attention 路由与 Value"),("mlp","MLP 门控与 Up 内容")):
        l=c[f"candidate_{mode}_layer"]
        q=[]
        for phase,label in FAMILIES:
            a=phases[phase]["atlas"]
            for t,r in a["metrics"].items():
                x=r[l-1][mode]
                q.append((label,t,a["n"],*(f"{v:.4g}" for v in x["projected"]),
                         f"{x['max_residual']:.2g}",f"{x['norm_remainder_median']:.4g}"))
        a=table(("样本","轮","n","因子 RMS","内容 RMS","交互 RMS","最大相对恒等式残差","归一化剩余 RMS"),q)
        tables[mode]=a
        doc(root,"ATTENTION_ROUTING.md" if mode=="attention" else "MLP_GATING.md",title,
            [att_text if mode=="attention" else mlp_text,a,
             "比值筛选只使用开发集输入与原生计算。完整逐层逐头结果见 ATLAS_*.json，响应和行为证据分别见 [FUNCTIONAL_RESPONSE.md](FUNCTIONAL_RESPONSE.md) 与 [CAUSAL_SWITCH.md](CAUSAL_SWITCH.md)。"])
    response=[]
    for phase,label in FAMILIES:
        for delta,x in phases[phase]["causal"]["response"].items():
            response.append((label,delta,x["gain"][0]["n"],interval(x["gain"][0]),interval(x["gain"][1]),
                             *(interval(z) for z in x["cosine"])))
    response_table=table(("数据","δ","n","K1 增益","K2 增益","K1→2 响应余弦","K2→3","K3→4"),response)
    scaling=load(o/"RESPONSE_CONTROLS.json")
    rt=[]
    for phase,z in scaling["rows"].items():
        for metric in ("same_delta","half_delta","same_relative_delta","norm_matched_H0_vs_H1"):
            rt.append((phase,metric,len(z),f"{sum(q[metric]['cosine'] for q in z)/len(z):.4f}",
                  f"{sum(q[metric]['ratio'] for q in z)/len(z):.4f}"))
    response_control_table=table(("数据","控制","n","平均响应余弦","平均增益比"),rt)
    increments=s["increment"]
    inc_table=table(("留出","训练 n","K1 错 n","K2 纠正","基线 Brier","加入响应后","配对改善"),
      [(p,x["train_n"],x["n"],x["positives"],f"{x['baseline_brier']:.3f}",f"{x['extra_brier']:.3f}",interval(x["improvement"]))
       for p,x in increments.items()])
    doc(root,"FUNCTIONAL_RESPONSE.md","同一模型对相同有限扰动的响应",
        ["计算 R(H,δ)=F(H+δ)−F(H)。固定扰动在输入中点，用同一个种子和向量，绝对剂量取 K1 输入该 token 范数的 12%。"
         "兼容 token 布局的等义改写用真实 K1 差，裁去扰动位置之后的差并匹配全向量剂量。增益以整有效序列空间计算，方向由同题配对。",
         response_table,"对比相同幅度、半剂量、相对剂量和逐 token 范数匹配。范数匹配 H0 是对照反事实，实际自然异向量仍来自原生轨迹。",
         response_control_table,"FF 能产生不同响应是一条操作性因果结论。后续用开发固定正规化的只读诊断估计响应在置信 margin、范数、注意力熵和更新量之外的预测增量。",
         inc_table])
    swap=[]
    sub=[]
    target=[]
    for phase,label in FAMILIES:
        v=phases[phase]
        for t,cell in v["causal"]["predictions"].items():
            for mode,a in cell.items():
                factor=a["arms"]["factor"]
                a1=a["arms"]
                swap.append((label,t,mode,interval(factor["progress"]),interval(a["contrast"]["random"]),
                    interval(a["contrast"]["other_tokens"]),interval(a["contrast"].get("other_heads")),factor["ood2x"]))
                if t in ("1from2","2from1"):
                    cf=a["stratified_factor"]["tf"]
                    sub.append((label,t,mode,*[interval(cf[k]) if cf[k] is None else f"{cf[k]['n']} · {interval(cf[k])}" for k in ("WC","CW","CC","WW")]))
        for t,d in v.get("full_patch",{}).items():
            for mode,a in d.items():
                for k,x in a.items():
                    target.append((label,t,mode,k,x["margin_effect"]["n"],interval(x["margin_effect"]),
                                   x["correct_before"],x["correct_after"],x["rescues"],x["harms"]))
    swap_table=table(("样本","接收 from 供体","路径","因子方向","相对同剂量随机","相对位置位移","相对其他头","范数 >2×"),swap)
    target_table=table(("样本","方向","路径","成分","n","margin 变化","原生正确","交换后正确","救回","损伤"),target)
    doc(root,"CAUSAL_SWITCH.md","真实原生因子互换",
        ["自然接收方的前缀、被干预层和剩余层及 Loop 预算相同。Attention 在 o_proj 之前以完整 softmax 概率和相应 V 做析因，MLP 在 down 之前交换 G 或 U。"
         "换入的因子在供体中曾实际出现，混合本身是反事实。与随机和位移方向保持同剂量，下游保留原生 o_proj/down、norm、residual。其他层是真实替换的额外未配平对照。",
         swap_table,table(("样本","方向","路径","错→对","对→错","对→对","错→错"),sub),
         "上表为完整候选平均 LL 原生行为分层。真实首 token logit 与词表对齐不同，也有单独完整候选移植，在只改自然提示 token 后评分续写。",
         target_table,"预算匹配只在每对反事实同一个接收方内比较。开发双向门槛两个候选均失败，后续主数据检验失败范围。"])
    forms=[]
    for ph,items in fs.items():
        for m,q in items.items():
            restore="route" if m=="attention" else "gate"
            forms.append((ph,m,q["arms"]["drop"]["factor_loss"]["n"],interval(q["arms"]["drop"]["factor_loss"]),
                         interval(q["origin_specificity"]["matched_random"]),
                         interval(q["restore"][restore]),interval(q["first_token"]["rescue"]),q["correct"]))
    form_table=table(("数据","路径","n","写入删除后因子损失","相对等剂量随机损失","相应因子恢复的读出收益","首 token margin 恢复","原生/删除/恢复正确"),forms)
    opening=[]
    for mode,w in (("attention",al),("mlp",mw)):
        early=[fs[key][mode] for key in ("confirm_arc","confirm_external")]
        drops=[f"{x['arms']['drop']['factor_loss']['mean']:.3f}" for x in early]
        boots=[interval(x["origin_specificity"]["matched_random"]) for x in early]
        opening.append(f"删除 `{w}` 后，自然因子方向损失依 ARC/OpenBookQA 顺序为 {' / '.join(drops)}，相对同剂量随机的差为 {' / '.join(boots)}。")
    origin_narrative="".join(opening)+"这些量是相对自然差方向的投影单位。方向变化、正交余量和行为变化仍可分离。"
    doc(root,"CROSS_LOOP_ORIGIN.md","上一轮真实写入与下一轮共享计算",
        [f"形成诊断使用标签隔离后的 16 道开发题遍历 48 个真实写入。注意力候选最终冻结为 K1 `{al}` → K2 L{ar} 的注意力图，MLP 候选为 K1 `{mw}` → K2 L{ml} 的整层 gate。去除在原生分支输出归一化后、加回残差前完成；真实前缀和后缀仍执行。",
         origin_narrative,form_table,
         "去除会同时改变多个后续中介。各支单独恢复在被去除背景下的效果是条件效应。保留写入并替换删除背景实际产生的中介属于反方向控制。Q、K、V、Gate、Up 子项、等剂量扰动、其他写入和两倍范数阈值逐题保留。用该诊断不得把全链归因给一个自然中介。"])
    gen_table=behavior_table(phases)
    transitions=[]
    for phase,label in FAMILIES:
        for name,caption in PROT:
            tr=phases[phase]["behavior"][name]["transitions"]
            for i,z in enumerate(tr,1):
                transitions.append((label,caption,f"K{i}→K{i+1}",*[z.get(k,0) for k in ("WC","CW","CC","WW")]))
    doc(root,"NATIVE_COMPUTATION_ATLAS.md","欧若原生完整轨迹与行为图谱",
        [nat_para,gen_table,table(("样本","协议","转移","错→对","对→错","对→对","错→错"),transitions),
         "自由生成使用贪心全词表、无受限候选解码，最多 12 个新 token 或首个完整行。按第一非空行精确匹配全答案，拒绝半答案和并列。不解析计为错。LL 独立测量原始文本完整候选续写 token，总分与平均分分开。选项字母使用带显式选项的另一提示。轨迹图谱与因果分析采用标准开放提示。"])
    sem=[]
    for label,v in s["semantics"].items():
        for donor,modes in v.items():
            for mode,arms in modes.items():
                a=arms["factor"]
                sem.append((label,donor,mode,a["n"],interval(a["relative_random_margin"]),
                           a["target_hits"],a["donor_hits"],a["native_hits"],a["random_hits"]))
    doc(root,"GENERALIZATION.md","独立数据与结构中的因果范围",
        ["OpenBookQA 96 是本轮首次执行的独立任务族。固定快照排除 432 个历史 ID 和重复/近重复条目。ARC Train 96 是对开发参与集合的留出，当前项目有更早的 ARC 全集接触。开发/确认保留同一哈希排序，不按正确性筛题。"
         "结构两族各 24 组使用新实体和规则随机种子，以及新末尾指令。跨世界匹配包括同桥异答案、同答案异桥和变换来源但保持桥；这些配对与真实新答案条件均在运行前从结构规则指定。",
         gen_table,table(("结构","供体关系","路径","n","受体目标 margin−随机","目标首选","供体首选","原生目标","随机目标"),sem),
         "正确定义用受体规则重算：同答案异桥时，以受体规则处理供体桥的答案为预期。同桥异答案和变来源同桥时，预期保持受体的原目标。合成四世界的 score 是候选首 token 诊断，完整生成/LL 另有报告。"])
    concrete=[]
    for phase,label in FAMILIES[:2]:
        ph=phases[phase]
        val=ph["causal"]["response"]["fixed"]["cosine"]
        ll=ph["behavior"]["tf"]["transitions"][0]
        gn=ph["behavior"]["gen"]["transitions"][0]
        concrete.append(f"{label} 的同 δ 响应 K1→K2 余弦 {val[0]['mean']:.3f}，K3→K4 升至 {val[-1]['mean']:.3f}。"
            f"K1→K2 完整候选纠正 {ll.get('WC',0)}，回退 {ll.get('CW',0)}；严格自由生成纠正 {gn.get('WC',0)}，回退 {gn.get('CW',0)}。")
    report=["## 结论",
        "Ouro 的共享权重与循环调用本身保持不变。真实状态改变后，同一个算子对相同有限扰动给出不同方向和增益，说明变化超过了直接把不同状态送入 lm head。开发候选能改变下一轮内部计算，但两条互换都未通过双向、同剂量对照的判据。当前证据能定位部分自然计算变化的来源，尚不能把 K1 错误到深轮答对归因给某个可重复的路由或 gate 开关。",
        "".join(concrete), "## 1. 哪些自然状态改变了计算",nat_para,response_table,response_control_table,
        "## 2. 信息积累、路由与答案边界能区分到什么程度",
        "因子变化在投影及原生后置归一化前后均可见，直接运行有限扰动证明响应随状态而变。因子的互换不能单独和双向地复现原生行为。开发筛选按分量相对实际差的比例，抵消会抬高比例，所以只对冻结候选和剂量给出阴性结论。信息积累与决策边界依然是行为变化的竞争解释，H1、H2、H3 在本设计中并不互斥。",
        inc_table,"## 3. 哪些隐藏状态成分控制了变化",att_text,mlp_text,
        "## 4. 形成这些成分的前轮写入",
        f"独立形成诊断锁定 K1 的 `{al}` 和 `{mw}` 两个实际写入，同一套冻结测试也执行 K2→K3。"
        "这两个来源是删除实验定位到的候选，报告与随机、位移、其他写入之间的配对效应；mediator-only、清洁恢复与删写入的实际组合都已记录。",
        origin_narrative,form_table,
        "## 5. 能否因果解释纠正",
        "下面的原生准确率使用不同评测协议分别计数。因果互换的完整候选效应另表，不能从整词表方向或首 token 的变化推出自然生成答案已修复。",
        gen_table,target_table,
        "## 6. 可以推广什么",
        "在 ARC 和排除历史记录的 OpenBookQA 比较同一锁定候选，并在新生成表格/两跳结构检验受体规则预测。强主张需要读出、下游响应和任务行为同时复现。各协议、K2→K3/K3→K4 和对照详见对应附件。",
        "## 7. 相对既有研究的增量",
        "近邻已经在训练得到的图任务网络上对 Attention pattern 做干预，也已有 Ouro 的自然状态/注意力回拨与激活互换。这里的可复核增量是冻结自然样本上的 A/V 与 SwiGLU 全层精确分解，同一扰动的跨状态响应，以及既定门槛下向前追踪真实写入的反事实范围。"
        "现有结果支持在冻结 Ouro-1.4B 上的具体边界结论，无普适算法切换或隐藏语言推理链主张。[近邻核查](NOVELTY_AUDIT.md)",
        "## 产物",
        "原始预测 SHA、赋值后原生运行、逐题/世界组记录、完整 FP32 代表轨迹、环境、统计和图均在本目录。摘要中的双侧 95% 区间按题目或结构组重抽样；开发模型推进门槛另有多重比较修正。"]
    doc(root,"REPORT_ZH.md","Ouro-1.4B 的自然状态如何改变共享算子",report)
    doc(root,"CLAIM_EVIDENCE_TABLE.md","主张与证据",[
        table(("主张","证据","判定范围"),[
        ("同一输入下每轮共享算子一致","RUNTIME 的 FP32/BF16 逐位审计","在 no-cache、固定 mask 和 RoPE 下通过"),
        ("自然状态改变相同扰动的实际作用","完整原生前向与等/半/相对剂量，逐 token 范数配平","跨两个自然任务定位方向与增益，非只看激活"),
        ("A/V 和 SwiGLU 自然变化可作代数分解","ATLAS，四轮全物理层与 heads","投影前后恒等式，归一化留有非线性余量"),
        ("候选 A 或 gate 是纠正的通用开关","预登记双向交换对控制的门槛","未通过开发门槛，各输入保留全量失败范围"),
        ("上一轮写入重塑下一轮的局部因子","16 题扫描固定写入，独立形成诊断，配对控制","按任务与 Loop 查表，条件中介不等同独占解释"),
        ("新结构复用受体任务变量","组级同桥异答案等 donor 规则与匹配随机","见全体组结果，避免单例和先选成功"),
        ("模型新执行某个内部算法","行为、相同输入 parity、结构和互换","证据未达到该层面")
      ])])


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("results",type=Path)
    a.add_argument("directory",type=Path)
    v=a.parse_args()
    v.directory.mkdir(exist_ok=True)
    reports(v.results,v.directory)

