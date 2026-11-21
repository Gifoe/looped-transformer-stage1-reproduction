"""Verify frozen cohorts, native K2 deletions, and margin-matched statistics."""

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import scipy
import torch
import transformers

from native import MODEL_HASH,REV,load,save,sha
from nec_deck import DOMAINS,build,summary
from nec_match import CALIPER,choose,enroll,matches
from nec_stats import ARMS,summarize
from nec_test import auditors
from rule import MAX_NEW,parse

R = Path("results")


def number(z):
    assert 0<len(z["tokens"])<=MAX_NEW and parse(z["text"])==z["number"]


def verify_native():
    d,n,z = [load(R/name) for name in ("DECK.json","NATIVE.json","NATIVE_SUMMARY.json")]
    assert build()==d and sha(R/"DECK.json")==n["deck_sha256"]
    for a,b in zip(d["rows"],n["rows"],strict=True):
        assert a=={k:v for k,v in b.items() if k!="native"}
        assert len(b["native"]["original"])==2
        for a0 in b["native"]["original"]:
            number(a0)
    assert summary(n["rows"])==z
    v = dict(status="PASS",model_revision=REV,expected_weights_sha256=MODEL_HASH,
             weights_sha256=os.environ["OURO_ACTUAL_SHA256"],gpu=os.environ["OURO_GPU"],torch=torch.__version__,
             scipy=scipy.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
             prompts=len(d["rows"]),deck_sha256=sha(R/"DECK.json"),native_sha256=sha(R/"NATIVE.json"))
    assert v["weights_sha256"]==MODEL_HASH
    return v,n["rows"],z


def verify():
    v,rows,z = verify_native()
    if not z["gate"]:
        return v
    m,c,r = [load(R/x) for x in ("MARGINS.json","COHORT.json","NEC_RAW.json")]
    assert m["native_sha256"]==c["native_sha256"]==v["native_sha256"]
    assert c["margins_sha256"]==sha(R/"MARGINS.json") and r["cohort_sha256"]==sha(R/"COHORT.json")
    chosen = enroll(rows)
    assert c["cases"]==[x["id"] for x in chosen]==[x["id"] for x in m["rows"]]==[x["id"] for x in r["rows"]]
    match = matches(m["rows"])
    assert c["matched"]==match and c["audit"]==auditors(chosen)
    inspected = {i for d in c["audit"].values() for g in d.values() for i in g}
    for a,b,u in zip(chosen,m["rows"],r["rows"],strict=True):
        assert {k:a[k] for k in ("id","domain","kind")}=={k:b[k] for k in ("id","domain","kind")}
        assert len(b["choices"])==len({x["foil"] for x in b["choices"]})
        candidates = {a["foil"]} if a["kind"]=="repair" else set(a["example"]["original"]["values"])-{a["target"]}
        for w in b["choices"]:
            assert w["foil"] in candidates and math.isfinite(w["margin"]) and w["margin"]>=0
            assert w["position"]>=0 and w["target_token"]!=w["foil_token"]
        assert b["selected"]==choose(b["choices"])
        if a["kind"]=="stable" and b["selected"] is not None:
            a["foil"]=b["selected"]["foil"]
        assert {k:u[k] for k in a}==a
        assert set(u["pred"])==set(ARMS) and u["pred"]["identity"]==u["native"]["original"][1]
        for prediction in u["pred"].values():
            number(prediction)
        assert ("audit" in u)==(u["id"] in inspected)
        if "audit" in u:
            q=u["audit"]
            assert q["checks"]==18 and all(0<=x<.001 for k,x in q.items() if k!="checks")
    for d,pairs in match.items():
        assert all(0<=p["absolute_gap"]<=CALIPER for p in pairs)
        assert len(pairs)==len({p["repair"] for p in pairs})==len({p["stable"] for p in pairs})
    assert json.loads(json.dumps(summarize(r["rows"],match)))==load(R/"NEC_SUMMARY.json")
    v.update(margins_sha256=c["margins_sha256"],cohort_sha256=r["cohort_sha256"],raw_sha256=sha(R/"NEC_RAW.json"),
             peak_allocated_mib=r["peak_allocated_mib"],runtime_s=r["elapsed"])
    return v


def plot(z):
    fig,axes=plt.subplots(1,2,figsize=(8.8,4),sharey=True)
    colors=("#3A688A","#A55243")
    names=("Directed reversion","Matched specificity")
    for ax,name,key in zip(axes,names,("directed","specificity"),strict=True):
        for i,d in enumerate(DOMAINS):
            x=z["domains"][d][key]
            if x is not None:
                mean=100*x["mean"]
                lo,hi=[100*a for a in x["interval"]]
                ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt="o",ms=5,capsize=4,color=colors[i])
        ax.axvline(0,color="0.6",linewidth=.7)
        ax.set_title(name,loc="left")
        ax.set_yticks(range(len(DOMAINS)),[d.title() for d in DOMAINS])
        ax.set_xlabel("Effect (percentage points)")
        ax.spines[["right","top"]].set_visible(False)
        ax.set_ylim(-.6,1.6)
    fig.tight_layout()
    p=R/"NATIVE_K2_DELETION.png"
    fig.savefig(p,dpi=180,bbox_inches="tight")
    plt.close(fig)
    with Image.open(p) as i:
        assert i.width>1000 and i.height>450
        i.verify()
    return sha(p)


def interval(value):
    if value is None:
        return "无法估计"
    lo,hi=[100*x for x in value["interval"]]
    return f"{100*value['mean']:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def render(v):
    n=load(R/"NATIVE_SUMMARY.json")
    d=["# 原生 K2 注意力交互项删除", "",
       "固定 Ouro-1.4B Base。每次生成先计算同一前缀的干净 K1、K2，再从重新运行的 K2 第 12–17 层实时注意力投影中减去两轮的路由–内容乘积位移。其他注意力写入、归一化和 MLP 继续随受扰状态计算。几何对照匹配每个位置的投影位移范数与相对干净 K2 投影的径向分量。", "",
       "| 格式 | 题数 | 错→错 | 错→对 | 对→错 | 对→对 | 精确纠正 |",
       "|---|---:|---:|---:|---:|---:|---:|"]
    for k in DOMAINS:
        a=n[k]
        cells=" | ".join(str(a["transitions"][i]) for i in ("00","01","10","11"))
        d.append(f"| {k} | {a['n']} | {cells} | {a['repairs']} |")
    if not n["gate"]:
        d.extend(["","精确自然纠正样本数未达到预设门槛。"])
    else:
        z=load(R/"NEC_SUMMARY.json")
        d.extend(["", "| 格式 | 题组 | 全部 n | 定向 n | 真实删除，定向 / 任意错 | 几何删除，定向 / 任意错 | 零删除，定向 / 任意错 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
        for k,q in z["domains"].items():
            for g,w in q["arms"].items():
                hits=" | ".join(f"{w['counts'][a]['foil']} / {w['counts'][a]['error']}" for a in ARMS)
                d.append(f"| {k} | {'自然纠正' if g=='repair' else '原生稳定'} | {w['n']} | {w['foil_n']} | {hits} |")
        d.extend(["", "自然纠正组的定向终点为同题 K1 实际生成的原错表值。稳定组为干净 K2 最接近的其他表值。按首次数字分歧的干净 K2 logit 差做无放回匹配。", "",
                  "| 格式 | 纠正组额外 / 减少 | 纠正组定向效应（百分点） | 匹配数 | 分差均值 / 最大 | 匹配特异性（百分点） | 定向效应同时上界（百分点） |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
        for k,q in z["domains"].items():
            p,b=q["directed"],q["balance"]
            gap=f"{b['mean']:.2f} / {b['max']:.2f}" if b["n"] else "无匹配"
            d.append(f"| {k} | {p['positive']} / {p['negative']} | {interval(p)} | {b['n']} | {gap} | {interval(q['specificity'])} | {100*p['upper_simultaneous']:.2f} |")
        d.extend(["", "上表是四项主比较的同时区间。下表为描述性逐项 95% 区间。", "",
                  "| 格式 | 纠正组任意错 | 稳定组任意错 | 稳定组定向 |", "|---|---:|---:|---:|"])
        for k,q in z["domains"].items():
            d.append(f"| {k} | {interval(q['repair_error'])} | {interval(q['stable_error'])} | {interval(q['stable_foil'])} |")
        audit=z["audit"]
        d.extend(["", f"按预设标准，原生路径依赖判据{'通过' if z['supported'] else '未通过'}，10 个百分点的定向效应{'可排除' if z['excludes_ten'] else '尚不能排除'}。",
                  f"固定 {audit['n']} 道题核验 {audit['checks']} 次现场删除，实时投影差误差 {audit['subtraction']:.3g}，因子等式误差 {audit['factorization']:.3g}，后缀独立重算误差 {audit['suffix']:.3g}。零删除的全体完整生成与干净 K2 一致。", "",
                  "![第 12–17 层交互项删除的定向作用](results/NATIVE_K2_DELETION.png)", "",
                  "此前联合交换的乘积消融未达到确认门槛。本轮以原生 K2 为起点，并保留受扰状态下实时重算的注意力。前序结果见[注意力因子交换](../ouro_lookup_av_20261121/REPORT_ZH.md)。"])
        v["figure_sha256"]=plot(z)
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__=="__main__":
    v=verify()
    render(v)
    save(R/"VERIFICATION.json",v)
    print(v,flush=True)
