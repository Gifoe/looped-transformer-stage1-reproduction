"""Render the Chinese report from the verified result tables."""
import argparse
from pathlib import Path

import numpy as np

from analyze import END, LABELS, load, paired, sha, winner


def ci(v, nd=3, scale=1):
    return f"{v['mean']*scale:+.{nd}f} [{v['ci'][0]*scale:+.{nd}f}, {v['ci'][1]*scale:+.{nd}f}]"


def append_table(lines, headings, rows):
    lines.extend(("", "| "+" | ".join(headings)+" |", "| "+" | ".join(["---"]*len(headings))+" |"))
    lines.extend("| "+" | ".join(map(str,row))+" |" for row in rows)
    lines.append("")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("report",type=Path)
    opt=p.parse_args()
    o=opt.out
    s=load(o/"SUMMARY.json")
    r=load(o/"REPORT_STATS.json")
    mf=load(o/"MANIFEST.json")
    d=s["decision"]
    phase=d["phase"]
    v=s["free"][phase]
    h=r["joint"]["FP32"]
    n=h["all"]["n"]
    val=r["free"]["n"]
    lines=["# Ouro-1.4B 的连续有效场与四世界组合",
           "",
           "## 连续化额外揭示了什么",
           "",
           "在同一真实物理层输入上，Attention→MLP 的二阶顺序项与两个分支的自身离散化项都影响单位幅度的状态预测。末层的顺序项还帮助预测原生 AM→BA 对组合分数的变化。它对局部误差的贡献得到验证，但单位幅度的整轮独立响应预测输给因果前缀 Anderson，原生 BA 的小剂量补偿只在 K2 有局部改善，满剂量在 K2 和 K4 均加大了偏差。因此证据支持有界的局部近似，以及独立于该近似的原生非加性组合。",
           "",
           "## 原生模型验证了哪些预测",
           "",
           f"全量原生确认包含 {n} 个预先保留的世界组，每种结构 96 组。四个 K1 起点共享接收态，来源与规则写入不重叠，联合起点的混合向量逐组恰为零。在两跳任务中移植的是第一跳 M 的规则，D 保持接收方设置。联合目标提前由任务规则决定，与接收方及两个单独供体的目标各不相同。K4 时两族平均混合 logit 均为正。下表保留五选一或三选一候选首 token 的严格最大分数和全分母。原生 intact 使用完整 H11 提示，不采用 patch。",
           ""]
    rows=[]
    for f in ("tableau","twohop","all"):
        for k,key in enumerate(END,2):
            z=h[f]["loops"][key]
            hits=z["hits"]
            rows.append((f"K{k} {f}",*(f"{hits[m]}/{h[f]['n']}" for m in ("joint","intact","random","unrelated")),ci(z["iq"])))
    append_table(lines,("范围","联合","完整提示","等量随机","无关位置","平均 Iq 与 95% CI"),rows)
    if (o/"NATIVE_confirmation.json").exists():
        truth=load(o/"NATIVE_confirmation.json")["rows"]
        meta={g["group"]:g for groups in mf["groups"]["confirmation"].values() for g in groups}
        novelty=[]
        for family in ("tableau","twohop"):
            available=emerged=0
            for run in truth:
                g=meta[run["group"]]
                if run["family"]!=family:
                    continue
                picks=list(map(winner,run["methods"]["native"][END[2]]["scores"]))
                y=g["targets"][-1]
                if y not in picks[:3]:
                    available+=1
                    emerged+=picks[3]==y
            novelty.append((family,emerged,available))
        parts="，".join(f"{family} {good}/{eligible}" for family,good,eligible in novelty)
        lines.append(f"事后按模型首选定义更严格的新选项。若前三条编辑轨迹均未首选 y11，K4 只有联合轨迹才首选 y11 的数量为 {parts}。主分析仍保留两个族各 96 组。")
    table_lanes = h["tableau"]["loops"][END[2]]["lane_hits"]
    bridge_lanes = h["twohop"]["loops"][END[2]]["lane_hits"]
    lines.append(f"K4 四条编辑轨迹按各自目标评分。接收、仅来源、仅规则、联合的正确数顺序为表格 {table_lanes}、两跳 {bridge_lanes}（每条 96）。")
    rows=[]
    for f in ("tableau","twohop"):
        q=h[f]["loops"][END[2]]
        for m in ("intact","random","unrelated"):
            rows.append((f,m,ci(q["joint_iq_vs"][m]),ci(q["joint_margin_vs"][m])))
    append_table(lines,("K4 结构","对照","联合减对照 Iq","联合减对照目标 margin"),rows)
    lines.append("随机方向匹配两种单独编辑的原支持和范数。无关位置把两种同剂量扰动投到同一个无关 token，联合总范数和空间支持未配平。")
    lines.extend([
       f"K4 合并配对命中差（百分点）对完整提示为 {ci(h['all']['loops'][END[2]]['joint_vs']['intact'],1,100)}，对等量随机为 {ci(h['all']['loops'][END[2]]['joint_vs']['random'],1,100)}，对无关位置为 {ci(h['all']['loops'][END[2]]['joint_vs']['unrelated'],1,100)}。两跳对随机的配对差为 {ci(h['twohop']['loops'][END[2]]['joint_vs']['random'],1,100)} 个百分点。",
       f"K4 BF16 复核命中表格 {r['joint']['BF16']['tableau']['loops'][END[2]]['hits']['joint']}/96、两跳 {r['joint']['BF16']['twohop']['loops'][END[2]]['hits']['joint']}/96。历史 63/96 来自 48 个旧结构组各两个目标和开放提示上的五候选诊断，完整提示是 56/96，两者全词表首 token 命中均为 0/96。历史样本用于回归核对。",
       f"K4 答案 token 的平均四世界状态 L2 在实际 norm 前后为 {h['all']['loops'][END[2]]['pre_post_norm_mean'][0]:.3f}→{h['all']['loops'][END[2]]['pre_post_norm_mean'][1]:.3f}。尾端 norm 同时改变坐标和尺度。后 norm 的主评分是固定目标 logit 减其他候选的均值，因此交互是四世界状态差的线性投影。",
       "",
       "## 独立自由推进",
       "",
       f"在 16 组开发数据中，原系数和两个共享阻尼均未通过预登记门槛。开发集选中原系数完整场。相对因果前缀 Anderson，K2、K4 配对误差改善依次为 {ci(d['controls'][END[0]]['gain'])}、{ci(d['controls'][END[2]]['gain'])} logit。按冻结规则在独立 {val} 组（每族 48）做失败范围复核；K1 初始化一次，后续各轮只用候选自身的状态。自由复核在两种任务族内留出新世界组；使用变更布局的 192 组用于独立原生确认。完整场从未使用真实的 K2/K3 边界来重置。",
    ])
    rows=[]
    for k,key in enumerate(END,2):
        q=v["all"][key]
        rows.append((f"K{k}",*(f"{q[m]['error']['mean']:.3f}" for m in ("full","base","taylor","prefix_anderson","zero")),
           f"{q['full']['positive_sign_exact']}/{q['full']['sign_evaluable']}",
           f"{q['full']['error']['p95']:.3f}",
           ci(r["free"]["stats"][key]["against"]["prefix_anderson"])))
    append_table(lines,("轮","完整场 MAE","无修正","离散 Taylor","前缀 Anderson","零响应","完整场符号","完整场 P95","相对 Anderson 改善与 95% CI"),rows)
    rows=[]
    for k,key in enumerate(END,2):
        x=r["free"]["stats"][key]["target_margin"]
        rows.append((f"K{k}",*(f"{x[m]['mae']['mean']:.3f} / {x[m]['wins']}" for m in ("full","base","taylor","prefix_anderson")),
                     v["all"][key]["native"]["target_correct"]))
    append_table(lines,("轮","完整场 margin MAE / 命中","无修正","离散 Taylor","前缀 Anderson","原生命中"),rows)
    lines.extend([
       f"结构分组的完整场 Iq MAE 列于 REPORT_STATS 与 SUMMARY。预测日志登记 {len(r['free']['failed'])} 个非有限方法批次，原生为 {len(r['free']['native_failures'])}。主分析按世界组作 4,000 次配对 bootstrap，表中改善定义为对照误差减候选误差。误差和 Iq 单位为 logit。K2 状态近似在局部相对无修正有所改善，但整轮对 Iq 的误差甚至高于零响应；状态误差的改善无法代替组合响应的预测能力。",
       "",
       "## 局部精度、数值误差与规范",
       "",
       "BF16 与精确提升至 FP32 各做三个小批，对齐 48 个残差写入与每种精度 24,870,912 个完整词表 logits。TF32、autocast、cache 均关闭，统一 math SDPA。终端 RMSNorm 的实数流采用径向一单位时段及逐坐标一单位时段，γ 全正，ε=1e−6。FP64 端点误差约 3.6e−15，FP32 实际 norm 相对差约 4.2e−8，BF16 约 2.35e−3。",
       "原生、Picard、因果前缀 Anderson、完整场在两组配对输入的前缀差均为零。全序列 Anderson 的最大前缀差约 0.352、0.041，共享混合权重读取了后缀残差；全序列 hybrid 差约 9.35e−5、1.72e−4。固定网格和只依赖前缀的混合权重用于后续比较。",
       "在开发的独立姿态上，α=1 的单 pair 相对残差端点中位误差依次为无修正 0.0721、自身 0.0524、顺序 0.0574、完整 0.0339、反号 0.0877、离散 Taylor 0.0228、前缀 Anderson 0.000428。α=1/8 的无修正约 0.00128、完整约 0.0000751、Taylor 约 0.0000511。此处是多层多轨迹描述性汇总，重复单位不能是 token 或层。",
    ])
    l=load(o/"LOCAL.json")["rows"]
    rows=[]
    for a in (.125,1.):
        for method in ("base","full"):
            for grid in (2,4,8,16):
                ks=[q for q in l if q["method"]==method and q["scheme"]=="rk4" and q["alpha"]==a and q["m"]==grid
                    and q["layer"] in (1,24)]
                if ks:
                    md=lambda k:np.median([z for q in ks for z in q[k]])
                    rows.append((a,LABELS[method],grid,f"{md('solver_error'):.3g}",f"{md('model_error'):.3g}"))
    append_table(lines,("α","场","RK4 m","相对同场 m=16 的 E_solver","相对离散映射的 E_model"),rows)
    rows=[]
    for depth,x in r["diagnostics"]["short"].items():
        rows.append((depth,*(f"{x[m]['pre_norm_error']['median']:.5f}" for m in ("full","base","taylor","prefix_anderson"))))
    append_table(lines,("真实前缀后的层数","完整","无修正","离散 Taylor","因果前缀 Anderson"),rows)
    lines.extend([
       "短段从真实 K2 前缀中第 6 层开始，段内自由推进。两层与四层的模型误差随长度增加，完整场的改进能延伸到四层，但仍输给前缀 Anderson。α、β 与 RK4 步长各有独立含义。完整场与第二积分设置、及 α 网格的逐例数值均在 LOCAL.json。高阶截断和后续传播是失败解释的候选，当前没有隔离各自的因果占比。",
       "",
       "## 原生算子顺序、剂量和桥接变量",
       "",
       "末层 L24 由开发集对 L12、L18、L24 的实际 swap 效应一次选出。每个 Loop 在独立 48 组上冻结方向和剂量，先保存预测，再把同一真实输入分别交给原生 AM 和 BA。预测保留真实前缀与原生末端 RMSNorm/head。各世界只在预定答案 token 注入方向。经理论反转的顺序项解释了部分 swap 变化，描述性最佳的局部拟合仍来自可执行相同 A/M 的因果 Anderson 对照。",
    ])
    rows=[]
    for k in (2,4):
        q=s["order"]["validation"]["loops"][str(k)]
        z=r["order"][str(k)]
        rows.append((f"K{k}",f"{z['absolute_swap_iq']['mean']:.3f}",
                     *(f"{q['order'][m]['error']['mean']:.4f}" for m in ("full","self","order","reverse","taylor","prefix_anderson")),
                     f"{z['margin']['full']['error']['mean']:.3f}",ci(q["full_vs_taylor"])))
    append_table(lines,("轮","实测平均 |ΔIq|","完整 MAE","自身","顺序","反号","Taylor","前缀 Anderson","完整 Δmargin MAE","完整相对 Taylor 的改善"),rows)
    rows=[]
    for k in (2,4):
        q=s["order"]["validation"]["loops"][str(k)]
        for x in (.25,.5,1.):
            z=r["order"][str(k)]["dose_gain"]
            rows.append((k,x,*(ci(z[a][str(x)]) for a in ("BA_comp","BA_random","BA_other")),
                         f"{q['dose_prediction'][f'BA_comp/{x}']['error']['mean']:.4f}"))
    append_table(lines,("轮","剂量","预测方向对 BA 的改善","随机方向对 BA","其他层方向对 BA","完整场补偿效应的预测 MAE"),rows)
    for k in (2,4):
        q=s["order"]["validation"]["loops"][str(k)]
        lines.append(f"K{k} 满剂量，完整方向较等范数随机改善 {ci(q['compensation_vs_random'])}，较其他层方向改善 {ci(q['compensation_vs_other'])}，较不加补偿的 BA 改善 {ci(q['compensation_vs_ba'])}。")
    lines.extend(["",
        "η=0.25 在 K2 对未补偿 BA、等范数随机和匹配其他层均有小幅改善，其中对 BA 的平均改善为 0.0121 [0.0058, 0.0192] logit。K4 未复现。剂量区间逐项报告，未作多重比较校正。满剂量推远了原本 BA 到 AM 的距离，两个 Loop 均出现这一结果。匹配随机方向的破坏更大，不能据此将 bracket 称为可恢复的语义介质。固定剂量曲线、AM 削弱、等支持随机和替代层均保留原始 logit 及预测。",
        "",
        "两跳 (b=M(s),y=D(b)) 使用只读的三候选真实 cabinet 词表头，位置在看到下游 D 之前以及答案位置。确认集 K1–K4 的 pre-D 命中为 "+"/".join(str(s["bridge"]["head"]["confirmation"][str(k)]["pre_D"]) for k in (1,2,3,4))+"（各 96）。同 b 不同 y 与同 y 不同 b 的单 token 移植各用同一 16 组，pre-D 的 K2/K3/K4 命中三元组为 "+", ".join(str(x["target_results"]) for x in s["bridge"]["swaps"] if x["roi"]=="pre_D")+"（接收、同 b 供体、不同 b 供体）。读出与移植没有形成稳定桥接机制证据。没有训练任务专用 probe，也没有断言存在隐藏语言推理链。",
        "",
        "## 自然任务、精度复核和时间重参数化",
    ])
    rows=[]
    for k,key in enumerate(END,2):
        for m,z in s["arc"][key].items():
            rows.append((k,LABELS[m],f"{z['eq']}/{z['n']}",f"{z['correct']}/{z['n']}",f"{z['native_correct']}/{z['n']}",
                         f"{z['readout_error']['median']:.4f}"))
    append_table(lines,("ARC 轮","方法","同原生选项","正确","原生正确","读出状态 L2 中位相对误差"),rows)
    lines.append("ARC-Challenge 取 validation 中此前未用于本轮方法调参及既有 embedding 阶段调参或确认的 96 题，整划分早前已被项目评分。主 ARC 是自然提示的原生多轮响应核对，和四世界受控语义实验分开。")
    if (o/"ARC_PERMUTATION.json").exists():
        x=load(o/"ARC_PERMUTATION.json")
        nat={q["item"]:q["methods"]["native"] for q in load(o/"ARC_NATIVE.json")["rows"]}
        gold={q["item"]:q["y"] for q in mf["arc_confirmation"]}
        assert x["arc_prediction_sha256"] == sha(o/"ARC_PREDICTIONS.json")
        rows=[]
        for k,key in enumerate(END,2):
            rs=[q for q in x["rows"] if q["item"] in nat]
            z=[(winner(q["scores"][str(k)]),winner(nat[q["item"]][key]["scores"]),gold[q["item"]]) for q in rs]
            if z:
                rows.append((f"K{k}",sum(a==b and a is not None for a,b,c in z),sum(a==c for a,b,c in z),sum(b==c for a,b,c in z),len(z)))
        append_table(lines,("ARC 固定轮换","原生内容一致","轮换后正确","相同题原顺序正确","有效 n"),rows)
        lines.append(f"选项轮换单独按内容重映射，{len(x['omitted'])} 题因模板或分词检查未进入这项探索性诊断，原 ARC 主结果分母保持 96。")
    rows=[]
    for k,key in enumerate(END,2):
        z=r["bf16"]["loops"][key]
        rows.append((k,*(f"{z[m]['error']['mean']:.4f}" for m in ("full","taylor","prefix_anderson"))))
    append_table(lines,(f"BF16（{r['bf16']['n']} 验证组）","完整场 Iq MAE","Taylor","前缀 Anderson"),rows)
    rows=[]
    for m,z in r["diagnostics"]["clocks"].items():
        rows.append((m,*(f"{z[c]['state_error']['median']:.3g} / {z[c]['signed_iq_abs']['median']:.3g}" for c in ("slow_fast","fast_slow"))))
    append_table(lines,("RK4 m","慢→快状态差 / |ΔIq|","快→慢状态差 / |ΔIq|"),rows)
    rows=[(key,f"{v['iq']['mean']:.4f}",f"{v['model_error']['mean']:.4f}") for key,v in r["diagnostics"]["beta"].items()]
    append_table(lines,("(βS, βR)","原生平均 |Iq|","完整场平均 E_response"),rows)
    lines.extend([
       "时钟结果是同一条件式开发局部字段的端点比较。网格加密使两种重参数化对固定参考时钟的差缩小。没有从数值时间推断模型内部事件时刻。β 验证使用事先指定的有限移植，在 L24 的实际 K2 输入上比较，数值趋势不能替代混合导数，也没有对整模型求嵌套二阶 JVP。",
       "",
       "## 成本与可复核范围",
    ])
    rows=[]
    for m in ("native","base","full","taylor","prefix_anderson"):
        z=s["timed_methods"][m]
        rows.append((LABELS[m],z["wall_s"]["n"],f"{z['wall_s']['median']:.3f}",f"{z['allocated_peak_bytes']['p95']/2**30:.2f}",
                     *(round(z[k]["median"]) for k in ("attention_calls","mlp_calls","jvp"))))
    append_table(lines,("方法","批次计时条目","中位 GPU 同步墙钟秒","显存峰值 P95 GiB","Attention 次数 中位","MLP 次数 中位","JVP 次数 中位"),rows)
    lines.extend([
      "成本来自 RTX 4090 的研究脚本，混合多种 batch 大小及短段/全轮场景，不能视作部署吞吐。按批访问和范围可在原始 timing 字段区分。所有实验使用冻结权重，没有训练或模型加速主张。代表组逐层逐 token 误差为 FP32 replay，完整宽度轨迹快照用 float16 分片。全组的状态混合范数保留逐 token 真值，方向的 24 维固定投影只作描述。预测、原生真值和运行日志分开保存。",
      "",
      "完整计算定义、资料和复现实验入口见 [THEORY.md](THEORY.md)、[BASELINE_AUDIT.md](BASELINE_AUDIT.md)、[NOVELTY_MATRIX.md](NOVELTY_MATRIX.md)、[PREREGISTRATION.md](PREREGISTRATION.md) 与 [README.md](README.md)。",
    ])
    pred_t=load(o/f"PREDICTIONS_{phase}.json")["timing"]
    live_t=load(o/f"NATIVE_{phase}.json")["timing"]
    rows=[]
    for family in ("tableau","twohop"):
        for name in ("native","full","taylor","prefix_anderson"):
            samples=[z for z in pred_t+live_t if z["run"]==f"{phase}/{family}/{name}"]
            if samples:
                def md(key):
                    return np.median([z[key] for z in samples])
                rows.append((family,LABELS[name],len(samples),f"{md('wall_s'):.3f}",f"{md('allocated_peak_bytes')/2**30:.2f}",
                    *(round(md(x)) for x in ("attention_calls","mlp_calls","jvp"))))
    append_table(lines,("独立自由推进","方法","4 组批次数","中位秒 / 批","峰值分配中位 GiB","A 次数","M 次数","JVP 次数"),rows)

    replay=o/"STATE_AUDIT.json"
    if replay.exists():
        subset=load(replay)["rows"]
        rows=[]
        for g in subset:
            for k in (2,4):
                key=f"K{k}/norm"
                role_map=dict(source=g["source"],rule=g["rule"],answer=[g["read"]])
                if "bridge" in g["roles"]:
                    role_map["bridge"]=g["roles"]["bridge"]
                for region,positions in role_map.items():
                    z=[]
                    for name in ("full","prefix_anderson"):
                        state=g["methods"][name][key]
                        err=np.array(state["relative_token_l2"])[:,positions].mean()
                        angle=np.array(state["mixed_cos"])[positions].mean()
                        z.append(f"{err:.3f} / {angle:.3f}")
                    rows.append((g["family"],k,region,*z))
        lines.extend(["","### 全宽代表组","",
            "每族取第一验证组重跑，在相同分支和物理边界配对状态。下表的方向为混合状态的余弦，相对误差先在四世界及角色 token 内平均。这里的两个组用于定位误差，无独立群体推断。"])
        append_table(lines,("族","轮","角色","完整场相对误差 / 余弦","前缀 Anderson 相对误差 / 余弦"),rows)
    opt.report.write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()

