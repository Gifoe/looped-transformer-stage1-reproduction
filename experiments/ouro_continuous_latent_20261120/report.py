"""Render the audit tables directly from released experiment summaries."""
import argparse
import json
import os
from pathlib import Path


def load(root, name):
    return json.loads((root / (name+".json")).read_text(encoding="utf-8"))


def write(dest, name, lines):
    (dest / (name+".md")).write_text("\n\n".join(lines)+"\n",encoding="utf-8")


def table(header, rows):
    lines=["| "+" | ".join(header)+" |", "|"+"|".join("---" for _ in header)+"|"]
    return "\n".join(lines+["| "+" | ".join(map(str,row))+" |" for row in rows])


def est(x, index=None, scale=1, precision=4):
    values=[x[k] if index is None else x[k][index] for k in ("mean","low","high")]
    a,b,c=(f"{v*scale:.{precision}f}" for v in values)
    return f"{a} [{b}, {c}]"


def endpoint(d):
    return f"{d['correct']}/{d['n']} ({d['correct']/d['n']:.2%})"


def docs(dest, root):
    s=load(root,"SUMMARY")
    f=load(root,"FREEZE")
    g=load(root,"GPU_TIMING")
    confirm=load(root/"confirmation","SUMMARY")
    confirm_checks=load(root/"confirmation","CHECKS")
    relative=Path(os.path.relpath(root,dest)).as_posix()

    def links(*names):
        return " · ".join(f"[{name}]({relative}/{name})" for name in names)

    geo=s["geometry"]
    ar=geo["arc/validation"]
    geo_rows=[]
    for k,d in geo.items():
        for pair,q in d["convergence"].items():
            if .5 in q["time"]:
                geo_rows.append((k,pair,q["n"],est(q,q["time"].index(.5)),est(q,-1),
                                 f"{max(q['mean']):.4f}"))
    acc_rows=[("native K1",endpoint(dict(correct=ar["native_k1"],n=ar["items"])),"—","—","—")]
    acc_rows += [(n,endpoint(t),t["rescue"],t["damage"],t["tied"]) for n,t in ar["endpoint"].items()
                 if t["n"]==299 and n not in ("linear","norm_linear","random")]
    write(dest,"CONTINUOUS_DYNAMICS",[
        "# 连续动力学",
        "距离为有效 token 全宽 L2 差除以比较状态的 L2 范数。组等权，方括号为配对组 bootstrap 95% 区间。"
        "ARC 图 3 始终用相同的 48 道面板题。下表验证集中的 m≥16 比较也仅覆盖这 48 道题，表中同时列出逐行分母。",
        table(("队列","比较","世界数","t=0.50","t=1","网格上最大均值"),geo_rows),
        "## 积分和精度对照",
        table(("先导组","比较","t=1"),[(co,k,est(v,-1)) for co,d in s["pilot_geometry"].items()
              for k,v in d["convergence"].items() if k in ("A32/rk4_8","A32/A32bf16")]),
        "## 全量 ARC-Challenge 验证集",
        "候选最高分精确并列判错。纠正及回退相对于正式 batch 的原生 K1。"
        f"独立预扫描的首最大候选与正式 batch 有 {s['arc_scan_vs_main_disagreements']} 道不同。"
        "高分辨率面板依预扫描冻结，按 K1/K2 的正确性分层，00/01/10/11 总数分别为"
        f" {', '.join(str(s['arc_native_strata'][k]) for k in ('00','01','10','11'))}。",
        table(("运行","准确率","纠正","回退","最高分并列"),acc_rows),
        "## 运行成本",
        f"正式轨迹同步分批墙钟合计 {s['profile_wall_s']:.1f} 秒，记录峰值 allocated {s['profile_peak_allocated']/2**30:.2f} GiB。"
        f"分批调用计数为 {s['profile_counters']}。因果各组真实调用次数和补充实验计数另存。",
        "固定 batch CUDA event 时长为一次预热后三次的中位数。"
        "A/B 行从 K1 开始，包含测量用路径采集，不含输出头。原生行从输入开始，含两轮和全词表输出头。",
        table(("运行","CUDA ms","峰值 allocated GiB"),[(k,f"{v['median_ms']:.2f}",
              f"{v['peak_allocated_bytes']/2**30:.2f}") for k,v in g["measurement"].items()]),
        links("trajectories.jsonl","runtime.jsonl","causal_execution.json","GPU_TIMING.json","SUMMARY.json")])

    stable=confirm["semantic"]
    sem_rows=[]
    order=[]
    for stage,summary in (("初版",s["semantic"]),("位置标签",stable)):
        for k,q in summary.items():
            if not k.startswith("twohop/new_structure/") or not q["ordered_b_before_y"]:
                continue
            method=k.rsplit("/",1)[-1]
            z=q["ordered_b_before_y"]
            order.append((stage,method,est(z,scale=100,precision=1),f"{z['both_detected']}/{q['items']}"))
            if method in ("native","A32","B32","linear","norm_linear","random"):
                for role,v in q["roles"].items():
                    sem_rows.append((stage,method,role.upper(),*[est(v,q["time"].index(t),100,1)
                                    for t in (0,.25,.5,.75,1)]))
    p=s["prediction"]
    cp=confirm_checks["prediction"]
    write(dest,"SEMANTIC_TRAJECTORIES",[
        "# 语义轨迹",
        "初版双跳读出将组内随机编号作为跨组分类标签。由于编号可重排，同一题面的语义类别无法唯一确定。"
        "该版本完整保留用于审计。复核版增加候选清单，并按可见题面定义 S（首表里的来源行）、"
        "M（首表第一行所指柜子在第二表的位置）、B（实际柜子在第二表的位置）、Y（选项清单里的对象位置）。"
        "两版提示不同，复核提示对所有构造轨迹及插值一致。两版的标签定义在开发与留出之间各自冻结。",
        "各表单位为百分数。均匀三分类机会为 33.3%。首次连续两个检查点都读对记为起点。"
        "顺序比例分母包含事件缺失的世界，也包含一个类别长时间不变而碰巧读对的世界。",
        table(("标签","新结构轨迹","变量","K1","t=0.25","t=0.50","t=0.75","终点"),sem_rows),
        table(("标签","轨迹","B 早于 Y（%）","两事件均出现"),order),
        "## 初版诊断配对与终局预测",
        "对应世界在 B 先于 Y 指标上的差异见下表。初版原生 K2 的真实正确性预测用 gold margin 作基线，"
        f"Brier {p['margin']['brier']:.4f}，再加初版桥标签的读出概率得到 {p['margin_plus_bridge']['brier']:.4f}，"
        f"配对下降 {est(p['paired_brier_reduction'])}。这一比较使用语义真值，初版编码问题限制解释。"
        "位置标签的复核版在同样的独立分组上用两个诊断模型评估。"
        f"仅 margin 的 Brier 为 {cp['margin']['brier']:.4f}，加入桥读出后的 Brier 为 {cp['margin_plus_bridge']['brier']:.4f}，"
        f"配对下降 {est(cp['paired_brier_reduction'])}。"
        "结构留出正确及错误世界的分层结果保存在各自汇总文件。",
        table(("初版配对","顺序差异（百分点）"),[(k,est(v,scale=100,precision=1))
             for k,v in s["semantic_checks"]["paired_order"].items()]),
        table(("位置标签配对","顺序差异（百分点）"),[(k,est(v,scale=100,precision=1))
             for k,v in confirm_checks["paired_order"].items()]),
        links("semantic_predictions.jsonl","native_readers.npz","SUMMARY.json",
              "confirmation/semantic_predictions.jsonl","confirmation/grounded_candidate_scores.jsonl",
              "confirmation/native_readers.npz","confirmation/SUMMARY.json","confirmation/CHECKS.json")])

    methods=("native","A16","B16")
    times=(.25,.5,.75)
    arms=("semantic","same_y_different_b","same_b_different_s","random","unrelated_token",
          "linear_semantic","semantic_minus_random","same_y_minus_random")
    ca=s["causal"]
    cc=confirm["causal"]
    write(dest,"CAUSAL_EVENTS",[
        "# 因果语义事件",
        "每组选用四个匹配世界。编辑只替换受体的最后一个 Answer 冒号状态，保留同一 batch 和真实后缀。"
        "目标依据供体桥值和受体规则。下表为新结构 12 组，开发 10 组结果及所有编辑向量实际范数见数据文件。",
        "目标 margin 为目标 logit 减最强其他候选 logit。初版随机及无关 token 臂与同规则供体匹配剂量。"
        "插值供体相对于自己路径的基线。方括号为组 bootstrap 95% 区间。",
        table(("轨迹","时间","臂","目标 margin 改变","目标胜出"),[
            (m,t,arm,est(q),f"{q['hits']}/{q['n']}") for m in methods for t in times for arm in arms
            for q in (ca[f"new_structure/{m}/{t}/{arm}"],)]),
        "## 每个供体的等剂量复核",
        "在原始提示上为三个供体各生成四个独立随机方向，以每个供体自己的编辑量匹配 L2 剂量，保留 BF16 后的实际剂量。"
        f"全量复核的相对剂量误差中位数为 {confirm_checks['relative_dose_error']['median']:.2%}，"
        f"最大值为 {confirm_checks['relative_dose_error']['max']:.2%}。"
        "三臂分别为 semantic（同规则换桥）、same_y（同最终答案换桥）及 same_b（同桥换来源）。",
        table(("轨迹","时间","臂","减真实基线","减本剂量四随机均值"),[
            (m,t,arm,est(q["edit_minus_clean"]),est(q["edit_minus_random"]))
            for m in methods for t in times for arm in ("semantic","same_y","same_b")
            for q in (cc[f"new_structure/{m}/{t}/{arm}"],)]),
        table(("轨迹","同答案换桥减随机，t=0.75 减 t=0.25"),
              [(m,est(cc[f"new_structure/{m}/same_y_late_minus_early"])) for m in methods]),
        links("causal.jsonl","causal_execution.json","confirmation/matched_dose_causal.jsonl","confirmation/SUMMARY.json")])


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("results",type=Path)
    p.add_argument("dest",type=Path)
    a=p.parse_args()
    a.dest.mkdir(exist_ok=True,parents=True)
    docs(a.dest,a.results)
