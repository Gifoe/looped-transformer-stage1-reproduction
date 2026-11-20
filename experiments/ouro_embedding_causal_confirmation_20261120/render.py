"""Build supplemental plots and the Chinese frozen-confirmation report."""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import FAMILIES, LABELS, METHODS, TYPES, read, summarize


def n(x):
    return f"{float(x):.4g}"


def md(head,rows):
    return "\n".join(["| "+" | ".join(head)+" |","|"+"|".join("---" for _ in head)+"|"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in rows])


def kv(d,k):
    return d.get(k,0)


def png(fig,out,no,name):
    for ext in ("png","pdf"):
        fig.savefig(out/f"figure_{no}_{name}.{ext}",dpi=250,bbox_inches="tight")
    plt.close(fig)


def plots(s,out):
    out.mkdir(exist_ok=True,parents=True)
    samples=[s["float32"]["loops"]["K2"]["causal"][f]["source_rule"] for f in FAMILIES]
    keys=("to_target","to_other","unchanged","tie")
    fig,ax=plt.subplots(figsize=(7,3.5),constrained_layout=True)
    bottom=np.zeros(len(FAMILIES))
    for k,c in zip(keys,("#177580","#a85443","#acada8","#382e3c")):
        y=np.asarray([kv(d["native_counts"],k) for d in samples])
        ax.bar(FAMILIES,y,bottom=bottom,color=c,label=k.replace("_"," "))
        bottom+=y
    ax.set(ylabel="Intervention arms",xlabel="Structural cohort",
           title="Native K1 → K2 counterfactual outcomes")
    ax.legend(frameon=False,ncols=2)
    png(fig,out,1,"early_native_semantic_outcomes")

    fig,axs=plt.subplots(1,2,figsize=(9.2,3.7),sharey=True,constrained_layout=True)
    methods=[k for k in METHODS if k!="native"]
    arms=("source_rule","same_y","random","unrelated")
    colors=("#1d668b","#b45145","#64765a","#7b6a88")
    ys=np.arange(len(methods))
    for ax,dt in zip(axs,TYPES):
        q=s[dt]["loops"]["K2"]["causal"]["all_structure"]
        for j,(arm,c) in enumerate(zip(arms,colors,strict=True)):
            v=[q[arm]["methods"][m]["logit_rmse"]["median"] for m in methods]
            ax.scatter(v,ys+(j-1.5)*.12,label=arm.replace("_"," "),color=c,s=19)
        ax.set(title=dt,xlabel="Median logit response RMSE",yticks=ys,yticklabels=[LABELS[m] for m in methods])
    axs[0].invert_yaxis()
    axs[1].legend(frameon=False,fontsize=8)
    png(fig,out,2,"causal_response_controls")

    fig,axs=plt.subplots(1,2,figsize=(9.8,3.5),constrained_layout=True)
    q=s["float32"]["loops"]
    depths=("K2","K3","K4")
    for m,c in zip(("chart_anderson","chart","prefix_ridge","full_loop","branch_flow"),
                   ("#1d668b","#5796a4","#503c57","#77679a","#a85443")):
        v=[q[k]["causal"]["paired_depth_panel" if k=="K2" else "all_structure"]["source_rule"]["methods"][m]
           ["logit_rmse"]["median"] for k in depths]
        axs[0].plot(depths,v,color=c,marker="o",label=LABELS[m])
        v=[q[k]["natural"]["arc"][m]["endpoint"]["mean"] for k in depths]
        axs[1].plot(depths,v,color=c,marker="o",label=LABELS[m])
    axs[0].set(ylabel="Median causal logit RMSE",title="Same 24 structural groups")
    axs[1].set(ylabel="Mean natural endpoint relative L2",title="32 ARC validation questions")
    axs[1].legend(frameon=False,fontsize=8)
    png(fig,out,3,"paired_loop_transfer")


def report(s,result,out):
    relative=Path(os.path.relpath(result,out)).as_posix()
    primary=s["float32"]["loops"]["K2"]["causal"]["all_structure"]
    task=primary["source_rule"]
    a=task["methods"]["chart_anderson"]
    controls=[]
    for dt in TYPES:
        for k,q in s[dt]["loops"].items():
            for arm,r in q["causal"]["all_structure"].items():
                d=r["native_counts"]
                g=r["methods"]["chart_anderson"]["counts"]
                controls.append((dt,k,arm,r["cases"],kv(d,"to_target"),kv(d,"to_other"),
                                 kv(d,"unchanged"),kv(d,"tie"),r["clean_correct_and_to_target"],
                                 kv(g,"matched_to_target"),kv(g,"matched_clean_correct_to_target"),kv(g,"spurious_flip")))
    cohorts=[]
    for f in FAMILIES:
        q=s["float32"]["loops"]["K2"]["causal"][f]["source_rule"]
        d=q["native_counts"]
        r=q["methods"]["chart_anderson"]["counts"]
        cohorts.append((f,q["cases"],kv(d,"to_target"),kv(d,"to_other"),kv(d,"unchanged"),kv(d,"tie"),
                        q["clean_correct_and_to_target"],kv(r,"matched_to_target"),
                        kv(r,"matched_clean_correct_to_target"),kv(r,"spurious_flip")))
    causal=[]
    contrasts=[]
    depths=[]
    for dt in TYPES:
        for k,q in s[dt]["loops"].items():
            for arm,r in q["causal"]["all_structure"].items():
                for m,values in r["methods"].items():
                    if m=="native":
                        continue
                    counts=values["counts"]
                    causal.append((dt,k,arm,LABELS[m],n(values["logit_rmse"]["median"]),
                                   n(values["direction_cos"]["median"]) if values["direction_cos"] else "—",
                                   f"{kv(counts,'matched_to_target')}/{kv(r['native_counts'],'to_target')}",
                                   f"{kv(counts,'final_correct')}/{r['cases']}",kv(counts,"spurious_flip"),
                                   kv(counts,"pred_tie")))
            r=q["causal"]["paired_depth_panel" if k=="K2" else "all_structure"]["source_rule"]
            d=r["native_counts"]
            depths.append((dt,k,r["cases"],kv(d,"to_target"),r["clean_correct_and_to_target"],
                           kv(r["methods"]["chart_anderson"]["counts"],"matched_to_target"),
                           *[n(r["methods"][m]["logit_rmse"]["median"]) for m in
                             ("chart_anderson","chart","full_loop","branch_flow","prefix_ridge")]))
            for method,d in q["causal"]["all_structure"]["source_rule"]["paired_chart_improvement"].items():
                contrasts.append((dt,k,LABELS[method],n(d["mean"]),f"[{n(d['ci'][0])}, {n(d['ci'][1])}]",d["groups"]))
    natural=[]
    for dt in TYPES:
        for k,q in s[dt]["loops"].items():
            for family,v in q["natural"].items():
                for m,d in v.items():
                    if m in ("ridge","persistence"):
                        continue
                    natural.append((dt,k,family,LABELS[m],d["n"],n(d["endpoint"]["mean"]),
                                    n(d["token_error"]["p95"]),n(d["direction"]["mean"]),
                                    n(d["max_logit_error"]["median"]),d["agree"],d["correct"],
                                    f"{d['natural_flips_hit']}/{d['natural_flips']}"))
    timing=[]
    for dt in TYPES:
        for measure,t in s[dt]["timing"].items():
            timing.append((dt,measure,n(t["wall_s"]["median"]),n(t["allocated_peak_bytes"]["p95"]/2**30),
                           n(t["attention_calls"]["median"]),n(t["mlp_calls"]["median"])))
    context=[]
    same_compare=[]
    for dt in TYPES:
        for k,q in s[dt]["loops"].items():
            new=q.get("intact_prompts",{})
            for family,v in new.items():
                context.append((dt,k,family,v["n"],kv(v,"donor_correct"),kv(v,"both_intact_prompts_correct"),
                                kv(v,"patched_recipient_reaches_target")))
            matched=q["causal"].get("same_y_scale_le4",{})
            all_same=q["causal"]["all_structure"]["same_y"]
            all_live=q["causal"]["all_structure"]["source_rule"]
            same_compare.append((dt,k,all_same["cases"],n(q["same_y_scale"]["median"]),
                                 n(q["same_y_dose_error"]["p95"]),
                                 n(all_live["native_output_effect"]["median"]),
                                 n(all_same["native_output_effect"]["median"]),
                                 matched.get("same_y",{}).get("cases",0)))
    r=task["native_counts"]
    ar=a["counts"]
    intact_k2=s["float32"]["loops"]["K2"].get("intact_prompts",{})
    intact_n=sum(kv(q,"both_intact_prompts_correct") for q in intact_k2.values())
    intact_hits=sum(kv(q,"patched_recipient_reaches_target") for q in intact_k2.values())
    stronger=task["paired_chart_improvement"]["prefix_ridge"]
    deploy=s["bfloat16"]["loops"]["K2"]["causal"]["all_structure"]["source_rule"]
    deploy_gap=deploy["paired_chart_improvement"]["prefix_ridge"]
    ci=a["ratio_to_target"]
    precision=(f"按来源组重采样的命中率区间为 [{n(ci['interval'][0])}, {n(ci['interval'][1])}]，"
               f"涉及 {ci['eligible_groups']} 个原生目标改选组。" if ci and ci["eligible_groups"]>1 else
               "可判目标改选来源组过少，无法据此估计稳定命中率。")
    if ci and ci["eligible_groups"]<10:
        precision+=" 此处事件稀少，区间仅描述当前面板上观察到的改选组。"
    nf=kv(r,"to_target")
    summary=[
        "# Ouro-1.4B 冻结反事实复核",
        "这是上一研究第四轮的独立验证扩展。求解器、权重、逆解迭代次数、两套 ridge 权重与比较口径预先冻结。"
        "上一轮 FP32 K1→K2 的 10 次原生改选命中全部来自 ARC，受控结构仅 0/2。"
        "本轮把受控结构单独作为主要检验，并补测真实 K3→K4。",
        "## 主要结果",
        f"48 个新受体组的来源/规则臂共 96 个。FP32 K1→K2 的原生模型改向目标 "
        f"{nf} 次、改向其他选项 {kv(r,'to_other')} 次、保持 {kv(r,'unchanged')} 次、并列 {kv(r,'tie')} 次。"
        f"Anderson 严格同时命中改选前后两个原生选项 {kv(ar,'matched_to_target')}/{nf}，"
        f"原生保持时误报 {kv(ar,'spurious_flip')}/{kv(r,'unchanged')}。"
        f"原生原先正确且改向目标 {task['clean_correct_and_to_target']} 次，Anderson 命中 "
        f"{kv(ar,'matched_clean_correct_to_target')} 次。{precision}",
        (f"完整受体和完整供体均答对的 {intact_n} 个早轮臂中，只写回一个角色后有"
         f" {intact_hits} 个受体改向目标。" if intact_k2 else ""),
        f"相对前缀 ridge，FP32 的按组平均 logit 响应 RMSE 改善为 {n(stronger['mean'])}，"
        f"95% 区间 [{n(stronger['ci'][0])}, {n(stronger['ci'][1])}]。"
        f"BF16 对应区间 [{n(deploy_gap['ci'][0])}, {n(deploy_gap['ci'][1])}]。",
        md(("新提示族","臂","原生改向目标","改向其他","保持","并列","原先正确且改向目标",
            "双端命中目标改选","其中原先正确","保持时误报"),cohorts),
        "纯 Anderson 依赖给定的原生分支函数。结果描述对真实编辑响应的重建，不能识别唯一的连续中间轨道。"
        "结构目标是程序从供体来源和规则算出。预测器只读编辑后的隐藏状态，目标标签仅用于计分。",
        "## 数据与对照",
        "3×3 新组 12 个，双跳开放提示 18 个，双跳列选项新格式 18 个。按既定种子与布局验收筛选，"
        "评分前冻结。开放提示族复用了模板，世界和语义表有隔离。32 道 ARC 验证题仅用于自然轨迹，"
        "对上一版嵌入拟合及验证选题独立，这批题曾出现在项目历史整分区评分中。",
        "每个结构受体组包含来源、规则、两种等剂量随机、两种位置对照及两个 same-Y 臂。"
        "same-Y 从目标不变的非相关前缀替换生成自然供体，写回相同角色并匹配剂量。"
        "角色中的原始词义与期望答案保持。剂量缩放可能把状态移出自然分布，"
        "原始供体剂量、缩放和写回后实际剂量全部在 JSON 中。另有双臂缩放均≤4 的敏感性子集。",
        md(("精度","终点轮","same-Y 臂","缩放中位","实际剂量相对偏差 P95","语义臂输出 L2 中位",
            "same-Y 输出 L2 中位","缩放≤4 的 same-Y 臂"),same_compare),
        "原生和所有预测器从真实同一轮初态起步。每类每个轮界检查 48 个分支边界和全词表输出对齐。"
        "本轮 CAUSAL 的 K3、K4 仅用评分前从每族哈希选出的 8 个组。下表 K2 也只用这同一批 24 组。",
        "## 配对的跨轮外推",
        md(("精度","终点轮","臂","原生改向目标","原先正确且改向目标","Anderson 双端命中",
            "Anderson RMSE","Picard RMSE","整轮 RMSE","分支 RMSE","前缀 ridge RMSE"),depths),
        "## 原生事件和严格改选",
        "并列时不算换选或保持。双端命中要求原生及预测的两端均无并列且各自选项一致。"
        "保持时误报要求原生两端相同、预测两端均可判但选项不同。合计分母包含原生走错目标的情形。"
        "随机与位置对照沿用对应语义臂的计分目标，只有 same-Y 具有保持原结构真值的语义解释。",
        md(("精度","终点轮","干预","臂","原生改向目标","改向其他","保持","并列","原先正确且改向目标",
            "Anderson 双端命中","原先正确的双端命中","保持时误报"),controls),
        "## 完整供体提示对照",
        "在原批次重放完整供体提示，保留所有语义条件，再与只写入指定角色的受体比较。"
        "这是在看到早轮目标改选稀少后补充的诊断，仍用冻结组，未按输出筛选。"
        "最后一列以 clean 和完整供体都答对的臂为分母来理解，区分原生能力不足与单角色编辑未传递足够信息。",
        md(("精度","终点轮","样本族","供体臂","完整供体答对","两完整提示都答对","其中编辑受体到达目标"),context)
        if context else "",
        "## 响应向量和候选 logits",
        "RMSE 是预测器自身 clean→edited logit 差与原生差之间的误差。"
        "方向余弦只对原生全宽响应非零的样本计算。配对差按来源组重采样，正值表示 Anderson 的平均 RMSE 更低。",
        md(("精度","终点轮","干预","方法","RMSE 中位","方向余弦中位","目标改选命中","最终命中指定目标",
            "保持时误报","预测并列"),causal),
        md(("精度","终点轮","比较法","平均差","95% 组 bootstrap 区间","来源组"),contrasts),
        "## 自然状态与成本",
        "自然输出用留出题原生全宽状态、原生候选 logits 和语义真值分别计分。"
        "候选一致数排除任一侧并列。原生自然改选是起始轮与终点轮之间的选项变化。"
        "时间为固定批次同步测量，含候选向量场求解，峰值为该窗口的 GPU allocated。"
        "Anderson 每个物理分支重算 8 次原生函数，原生只算 1 次。相同调用数的其他场仍有实现开销差别。",
        md(("精度","终点轮","样本族","方法","题数","平均相对 L2","token P95","更新余弦均值","最大候选 logit 差中位",
            "同原生","答对","原生自然改选复现"),natural),
        md(("精度","方法","同步墙钟秒中位","峰值分配 GiB P95","Attention 中位","MLP 中位"),timing),
        f"[冻结清单]({relative}/FREEZE.json) · [汇总]({relative}/SUMMARY.json) · "
        f"[图 1]({relative}/figures/figure_1_early_native_semantic_outcomes.pdf) · "
        f"[图 2]({relative}/figures/figure_2_causal_response_controls.pdf) · "
        f"[图 3]({relative}/figures/figure_3_paired_loop_transfer.pdf) · "
        "[上一轮六份报告与七图](../ouro_continuous_embedding_20261120/README.md)",
    ]
    out.mkdir(exist_ok=True,parents=True)
    (out/"CAUSAL_CONFIRMATION_ZH.md").write_text("\n\n".join(summary)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("data",type=Path)
    p.add_argument("reports",type=Path)
    a=p.parse_args()
    import json
    s=summarize(a.data)
    (a.data/"SUMMARY.json").write_text(json.dumps(s,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42})
    plots(s,a.data/"figures")
    report(s,a.data,a.reports)
