"""Rebuild row sites, enrollment, dose accounting, and exact-answer statistics."""

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
from nec_match import choose,enroll
from route_deck import DOMAINS,build,summary
from route_sites import freeze
from route_stats import ARMS,DIRECTIONS,PRIMARY,summarize
from route_test import L,ROUTING,auditors
from rule import MAX_NEW,parse

R=Path("results")
NAME={"route":"路由","value":"内容 V","joint":"联合","route_control":"路由对照","value_control":"V 对照",
      "joint_control":"联合对照","route_full":"全剂量路由","joint_full":"全剂量联合","identity":"自身回放"}


def exact(z):
    assert 0<len(z["tokens"])<=MAX_NEW and parse(z["text"])==z["number"]


def verify():
    d,n,z=[load(R/k) for k in ("DECK.json","NATIVE.json","NATIVE_SUMMARY.json")]
    assert d==build() and n["deck_sha256"]==sha(R/"DECK.json") and z==summary(n["rows"])
    for a,b in zip(d["rows"],n["rows"],strict=True):
        assert a=={k:v for k,v in b.items() if k!="native"}
        assert len(b["native"]["original"])==2
        for x in b["native"]["original"]:
            exact(x)
    v=dict(status="PASS",model_revision=REV,expected_weights_sha256=MODEL_HASH,
           weights_sha256=os.environ["OURO_ACTUAL_SHA256"],gpu=os.environ["OURO_GPU"],torch=torch.__version__,
           transformers=transformers.__version__,scipy=scipy.__version__,cuda=torch.version.cuda,
           prompts=len(d["rows"]),deck_sha256=sha(R/"DECK.json"),native_sha256=sha(R/"NATIVE.json"))
    assert v["weights_sha256"]==MODEL_HASH
    if not z["gate"]:
        return v
    m,c,r=[load(R/k) for k in ("MARGINS.json","COHORT.json","ROUTE_RAW.json")]
    assert m["native_sha256"]==v["native_sha256"]==c["native_sha256"]
    assert c["margins_sha256"]==sha(R/"MARGINS.json") and r["cohort_sha256"]==sha(R/"COHORT.json")
    tok=transformers.AutoTokenizer.from_pretrained(Path(os.environ["OURO_MODEL_PATH"]),trust_remote_code=True,local_files_only=True)
    rows=enroll(n["rows"])
    for a,b in zip(rows,m["rows"],strict=True):
        assert {k:a[k] for k in ("id","domain","kind")}=={k:b[k] for k in ("id","domain","kind")}
        assert b["selected"]==choose(b["choices"])
        alternatives={a["foil"]} if a["kind"]=="repair" else set(a["example"]["original"]["values"])-{a["target"]}
        for q in b["choices"]:
            assert q["foil"] in alternatives and math.isfinite(q["margin"]) and q["margin"]>=0
            assert q["target_token"]!=q["foil_token"]
    cases,paired,eligible=freeze(tok,rows,m["rows"])
    assert c["matched"]==paired and c["eligible"]==eligible and c["audit"]==auditors(cases)
    assert c["cases"]==[q["id"] for q in cases]==[q["id"] for q in r["rows"]]
    audited={i for d in c["audit"].values() for g in d.values() for i in g}
    for x,y in zip(cases,r["rows"],strict=True):
        assert {k:y[k] for k in x if k!="ids"}=={k:a for k,a in x.items() if k!="ids"}
        assert set(y["pred"])==set(DIRECTIONS)
        for a,b in zip(DIRECTIONS,x["native"]["original"],strict=True):
            assert set(y["pred"][a])==set(ARMS) and y["pred"][a]["identity"]==b
            for pred in y["pred"][a].values():
                exact(pred)
        assert ("audit" in y)==(y["id"] in audited)
        if "audit" in y:
            q=y["audit"]
            assert q["checks"]==len(L)*len(DIRECTIONS)*len(ARMS)
            assert all(0<=x<.001 for k,x in q.items() if k!="checks")
    assert json.loads(json.dumps(summarize(r["rows"],paired)))==load(R/"ROUTE_SUMMARY.json")
    assert 0<=r["fallback"][0]<=r["fallback"][1] and r["fallback"][1]>0
    dose=r["dose"]
    assert set(dose)=={f"{d}/{kind}/{direction}/{arm}" for d in DOMAINS for kind in ("repair","stable")
                      if any((x["domain"],x["kind"])==(d,kind) for x in cases) for direction in DIRECTIONS for arm in ROUTING}
    for key,meter in dose.items():
        assert len(meter)==len(L)
        for request,plan,actual in meter:
            assert all(math.isfinite(x) and x>=0 for x in (request,plan,actual))
            assert plan<=request+1e-6
            if not key.endswith("_full"):
                assert actual<=plan+1e-6
    v.update(margins_sha256=c["margins_sha256"],raw_sha256=sha(R/"ROUTE_RAW.json"),cohort_sha256=r["cohort_sha256"],
             fallback=r["fallback"],runtime_s=r["elapsed"],peak_allocated_mib=r["peak_allocated_mib"])
    return v


def plot(z):
    fig,plots=plt.subplots(1,2,figsize=(10,3.9),sharey=True)
    keys=(*PRIMARY,"route_full","joint_full")
    labels=("Route","Value","Joint","Full route","Full joint")
    for ax,d in zip(plots,DOMAINS,strict=True):
        n=z["domains"][d]["counts"]["identity"]["repair"]["n"]
        actual=[]
        for k in keys:
            counts=z["domains"][d]["counts"]
            actual.append(100*counts[k]["repair"]["dual"]/n)
        placebo=[100*counts[k+"_control"]["repair"]["dual"]/n for k in PRIMARY]
        x=range(len(keys))
        ax.bar([i-.15 for i in x],actual,.28,label="Target and foil",color="#3A688A")
        ax.bar([i+.15 for i in range(len(PRIMARY))],placebo,.28,label="Unrelated control",color="#A55243")
        ax.set_xticks(x,labels,rotation=15)
        ax.set_title(d.title(),loc="left")
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["right","top"]].set_visible(False)
    plots[0].set_ylabel("Exact rescue and reversal (%)")
    plots[1].legend(frameon=False)
    fig.tight_layout()
    dst=R/"TWO_ROW_ROUTING.png"
    fig.savefig(dst,dpi=180,bbox_inches="tight")
    plt.close(fig)
    with Image.open(dst) as p:
        assert p.width>1000 and p.height>450
        p.verify()
    return sha(dst)


def interval(x):
    if x is None:
        return "无配对"
    a,b=[100*y for y in x["interval"]]
    return f"{100*x['mean']:+.1f} [{a:+.1f}, {b:+.1f}]"


def render(v):
    n=load(R/"NATIVE_SUMMARY.json")
    d=["# 查表自然纠正中的两行路由与内容", "",
       "冻结 Ouro-1.4B Base。在同一前缀上分别交换两轮的注意力分配和 V。路由只在表格完整可见后移动正确行与原错行数字位置之间的注意力质量。V 交换覆盖这两行数字位置的原生值向量。所有层的其余注意力和后续运算接受受扰输入并自由重算。", "",
       "| 格式 | 原表题数 | 精确自然纠正 | 原生两轮均正确 |", "|---|---:|---:|---:|"]
    for fmt in DOMAINS:
        t=n[fmt]
        d.append(f"| {fmt} | {t['n']} | {t['repairs']} | {t['transitions']['11']} |")
    if not n["gate"]:
        d.extend(["","自然精确纠正样本数未达到预设门槛。"])
    else:
        raw=load(R/"ROUTE_RAW.json")
        cohort=load(R/"COHORT.json")
        z=load(R/"ROUTE_SUMMARY.json")
        d.extend(["", "| 格式 | 干预 | 纠正 n | K1 救回 / K2 原错 / 双向 | 匹配稳定 n | K1 保持 / K2 替代 / 双向 |",
                  "|---|---|---:|---:|---:|---:|"])
        for d0,q in z["domains"].items():
            for a,b in q["counts"].items():
                display=" | ".join(f"{b[k]['n']} | {b[k]['rescue']} / {b[k]['reversal']} / {b[k]['dual']}" for k in ("repair","stable"))
                d.append(f"| {d0} | {NAME[a]} | {display} |")
        d.extend(["","稳定组的同时终点要求 K1 保持正答、K2 输出预先固定的替代表值。完整任意错和方向计数见逐臂 JSON。","",
                  "| 格式 | 干预 | 方向效应（百分点） | 匹配特异性（百分点） | 方向上界（百分点） |",
                  "|---|---|---:|---:|---:|"])
        for k,q in z["domains"].items():
            for a,v0 in q["tests"].items():
                t=v0["directed"]
                d.append(f"| {k} | {NAME[a]} | {interval(t)} | {interval(v0['specificity'])} | {100*t['upper']:.2f} |")
        supported="、".join(NAME[a] for a in z["supported"]) or "无"
        excluded="、".join(NAME[a] for a in z["excluded"]) or "无"
        d.extend(["",f"跨格式通过直接效应和稳定组特异性的条件为 {supported}。按预设 10 个百分点阈值可排除的匹配剂量条件为 {excluded}。","",
                  "| 格式 | 方向 | 条件 | 共同可行剂量 / 原请求 | 实际剂量 / 原请求 |",
                  "|---|---|---|---:|---:|"])
        for fmt in DOMAINS:
            for dr in DIRECTIONS:
                for arm in ROUTING:
                    want,common,used=[sum(x) for x in zip(*raw["dose"][f"{fmt}/repair/{dr}/{arm}"],strict=True)]
                    d.append(f"| {fmt} | {dr} | {NAME[arm]} | {100*common/want:.1f}% | {100*used/want:.1f}% |")
        audit=sum(x["audit"]["checks"] for x in z["domains"].values())
        fallback,allv=raw["fallback"]
        d.extend(["",f"实际上限依据每个受扰前缀的源行质量。全剂量臂按实时两行质量重新分配。匹配组按首次数字分歧且干净 K2 logit 差相差不超过两个单位选取。实际两种格式的匹配数分别为 {len(cohort['matched'][DOMAINS[0]])} 和 {len(cohort['matched'][DOMAINS[1]])}。",
                  f"共核验 {audit} 次物理注意力计算，所有自身回放恢复原生整答。V 控制 {fallback:,}/{allv:,} 个位置与头使用旋转回退。",
                  "", "![两行注意力分配和 V 的双向作用](results/TWO_ROW_ROUTING.png)",
                  "", "前序实验已经在跨表冲突中确认目标行 V 的整答传递作用。本轮比较在同一片表格中，仅沿自然 K1/K2 变化方向改变这一内容或读取方式。[前序冲突实验](../ouro_lookup_conflict_20261121/REPORT_ZH.md)"])
        v["figure_sha256"]=plot(z)
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__=="__main__":
    v=verify()
    render(v)
    save(R/"VERIFICATION.json",v)
    print(v,flush=True)
