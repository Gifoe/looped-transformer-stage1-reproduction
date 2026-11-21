"""Verify exact matched routing, reproduce choice and pairing, and report outcomes."""

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

from dest_deck import DOMAINS,build,summary
from dest_stats import ARMS,DIRECTIONS,summarize
from dest_test import auditors,prepare,selection
from native import MODEL_HASH,REV,load,save,sha
from nec_match import choose,enroll
from route_sites import freeze
from rule import MAX_NEW,parse

R=Path("results")
NAMES={"true":"原生目的行","redirect":"第三行","live_full":"全实时路由","identity":"共同参考自身回放"}


def exact(a):
    assert 0<len(a["tokens"])<=MAX_NEW and parse(a["text"])==a["number"]


def verify():
    d,n,e=[load(R/x) for x in ("DECK.json","NATIVE.json","NATIVE_SUMMARY.json")]
    assert d==build() and n["deck_sha256"]==sha(R/"DECK.json") and summary(n["rows"])==e
    for a,b in zip(d["rows"],n["rows"],strict=True):
        assert a=={k:v for k,v in b.items() if k!="native"}
        assert len(b["native"]["original"])==2
        for v in b["native"]["original"]:
            exact(v)
    env=dict(status="PASS",model_revision=REV,weights_sha256=os.environ["OURO_ACTUAL_SHA256"],expected_weights_sha256=MODEL_HASH,
             gpu=os.environ["OURO_GPU"],torch=torch.__version__,transformers=transformers.__version__,
             scipy=scipy.__version__,cuda=torch.version.cuda,prompts=len(d["rows"]),deck_sha256=sha(R/"DECK.json"),
             native_sha256=sha(R/"NATIVE.json"))
    assert env["weights_sha256"]==MODEL_HASH
    if not e["gate"]:
        return env
    m,p,c,r=[load(R/x) for x in ("MARGINS.json","PROJECTION.json","COHORT.json","DEST_RAW.json")]
    assert m["native_sha256"]==c["native_sha256"]==env["native_sha256"]
    assert p["margins_sha256"]==c["margins_sha256"]==sha(R/"MARGINS.json")
    assert c["projection_sha256"]==sha(R/"PROJECTION.json") and r["cohort_sha256"]==sha(R/"COHORT.json")
    tok=transformers.AutoTokenizer.from_pretrained(Path(os.environ["OURO_MODEL_PATH"]),trust_remote_code=True,local_files_only=True)
    rows=enroll(n["rows"])
    for a,b in zip(rows,m["rows"],strict=True):
        assert {k:a[k] for k in ("id","domain","kind")}=={k:b[k] for k in ("id","domain","kind")}
        allowed={a["foil"]} if a["kind"]=="repair" else set(a["example"]["original"]["values"])-{a["target"]}
        assert b["selected"]==choose(b["choices"])
        for x in b["choices"]:
            assert x["foil"] in allowed and math.isfinite(x["margin"]) and x["margin"]>=0
    rows,pair,eligible=freeze(tok,rows,m["rows"])
    assert pair==c["matched"] and eligible==c["eligible"]
    cases=[]
    for a,b in zip(rows,p["rows"],strict=True):
        assert a["id"]==b["id"] and len(b["dose"])==2
        for choice in b["dose"]:
            assert len(choice)==3 and all(math.isfinite(x) and x>0 for x in choice)
        assert selection(*b["dose"])=={k:b[k] for k in ("third","ratio","projected_match")}
        cases.append(prepare(a,b))
    assert auditors(cases)==c["audit"] and c["cases"]==[x["id"] for x in cases]==[x["id"] for x in r["rows"]]
    audit={i for x in c["audit"].values() for y in x.values() for i in y}
    for expected,actual in zip(cases,r["rows"],strict=True):
        assert {k:v for k,v in expected.items() if k!="ids"}=={k:actual[k] for k in expected if k!="ids"}
        assert set(actual["pred"])==set(DIRECTIONS)
        for dr,clean in zip(DIRECTIONS,expected["native"]["original"],strict=True):
            arms=actual["pred"][dr]
            assert set(arms)==set(ARMS) and arms["identity"]==clean
            for pred in arms.values():
                exact(pred)
        assert ("audit" in actual)==(expected["id"] in audit)
        if "audit" in actual:
            a=actual["audit"]
            assert a["checks"]==192 and all(0<=v<.001 for k,v in a.items() if k!="checks")
    assert json.loads(json.dumps(summarize(r["rows"],pair)))==load(R/"DEST_SUMMARY.json")
    env.update(raw_sha256=sha(R/"DEST_RAW.json"),margins_sha256=c["margins_sha256"],
               projection_sha256=c["projection_sha256"],cohort_sha256=r["cohort_sha256"],
               runtime_s=r["elapsed"],peak_allocated_mib=r["peak_allocated_mib"])
    return env


def plot(z):
    fig,axes=plt.subplots(1,2,figsize=(8.6,4.1),sharey=True)
    for ax,d in zip(axes,DOMAINS,strict=True):
        q=z["domains"][d]["counts"]
        arms=("true","redirect","live_full")
        for i,(k,c) in enumerate((("repair","#3A688A"),("stable","#A55243"))):
            ax.bar([j+.32*(i-.5) for j in range(len(arms))],
                   [100*q[k]["arms"][a]["dual"]/q[k]["n"] for a in arms],.31,
                   color=c,label=k.title())
        ax.set_xticks(range(3),["Native destination","Third row","Full live"])
        ax.set_title(d.title(),loc="left")
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["right","top"]].set_visible(False)
    axes[0].set_ylabel("Exact joint outcome (%)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    p=R/"DESTINATION_CONTROL.png"
    fig.savefig(p,dpi=180,bbox_inches="tight")
    plt.close(fig)
    with Image.open(p) as i:
        assert i.width>1000 and i.height>450
        i.verify()
    return sha(p)


def effect(x):
    if x is None:
        return "无样本"
    a,b=[v*100 for v in x["interval"]]
    return f"{100*x['mean']:+.1f} [{a:+.1f}, {b:+.1f}]"


def render(v):
    n=load(R/"NATIVE_SUMMARY.json")
    d=["# 查表纠正的全剂量目的行对照", "",
       "冻结 Ouro-1.4B Base。每个实际生成前缀上，用受体干净前向中四个值行的数字位置建立共同参考。按供体的原生两行分配，从同一源行数字扣除完全相同的向量。原生臂把这份量加到供体指定的目的行，第三行臂沿相同的数字权重加到预先固定的第三行。其他位置保持实时相对权重，并共同缩放到剩余质量。V 和后续运算保持实时计算。上一轮的全实时路由作为单独的诊断。", "",
       "| 格式 | 原始题数 | 精确纠正 | 两轮均正确 |", "|---|---:|---:|---:|"]
    for k in DOMAINS:
        q=n[k]
        d.append(f"| {k} | {q['n']} | {q['repairs']} | {q['transitions']['11']} |")
    if not n["gate"]:
        d.extend(["","两个格式未同时达到预设自然纠正数量。"])
    else:
        z=load(R/"DEST_SUMMARY.json")
        cal={x["id"]:x for x in load(R/"PROJECTION.json")["rows"]}
        raw=load(R/"DEST_RAW.json")["rows"]
        d.extend(["", "| 格式 | 题组 | 条件 | n | K1 正答 / K2 原错 / 同时 | K1 第三行 / K2 第三行 / 同时 |",
                  "|---|---|---|---:|---:|---:|"])
        for fmt,q in z["domains"].items():
            for kind,y in q["counts"].items():
                for a,x in y["arms"].items():
                    s=" / ".join(str(x[k]) for k in ("rescue","reversal","dual"))
                    t=" / ".join(str(x[k]) for k in ("third_rescue","third_damage","third_dual"))
                    d.append(f"| {fmt} | {kind} | {NAMES[a]} | {y['n']} | {s} | {t} |")
        d.extend(["","稳定题的同时事件指 K1 保持正答且 K2 输出预选替代值。第三行是同一题实际存在的另一行。匹配子集按干净前向、首个预测前缀的 OV 投影范数筛选，匹配区间为比值 0.8–1.25。后续生成前缀随干预而变化。", "",
                  "| 格式 | 自然纠正差（百分点） | OV 参考匹配差（百分点） | 稳定组配对特异性（百分点） | 方向校正上界 |",
                  "|---|---:|---:|---:|---:|"])
        for fmt,q in z["domains"].items():
            d.append(f"| {fmt} | {effect(q['direct'])} | {effect(q['projected'])} | {effect(q['specificity'])} | {100*q['direct']['upper']:.2f} |")
        d.extend(["",f"跨格式目的行特异性判据{'通过' if z['supported'] else '未通过'}，预设等效上界判据{'达到' if z['excluded'] else '未达到'}。"])
        for fmt in DOMAINS:
            s=[r for r in raw if r["domain"]==fmt and r["kind"]=="repair"]
            m=sum(r["projected_match"] for r in s)
            a,b=[sorted(cal[r["id"]]["ratio"][j] for r in s)[len(s)//2] for j in (0,1)]
            d.append(f"{fmt} 中 {m}/{len(s)} 道满足两个方向的 OV 参考剂量匹配。第三行与原生目标的参考比值中位数为 {a:.3f}（K1）和 {b:.3f}（K2）。")
        audit=sum(x["audit"]["checks"] for x in z["domains"].values())
        e={k:max(x["audit"]["maxima"][k] for x in z["domains"].values()) for k in ("debit","l2","probability","attention")}
        d.extend(["",f"核验 {audit} 次物理注意力计算。逐数字源扣减差 {e['debit']:.3g}，安装后的平方范数差 {e['l2']:.3g}，注意力概率总量差 {e['probability']:.3g}，OV 逐头复算差 {e['attention']:.3g}。全部自身回放与原生整答一致。",
                  "", "![同源移除、不同目的行的整答对照](results/DESTINATION_CONTROL.png)", "",
                  "上一轮通过全实时双向诊断发现强效应。本轮用共同路由参考构造精确扣减对照，另保留全实时臂。[前序两行路由研究](../ouro_lookup_routing_20261122/REPORT_ZH.md)"])
        v["figure_sha256"]=plot(z)
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__=="__main__":
    v=verify()
    render(v)
    save(R/"VERIFICATION.json",v)
    print(v,flush=True)
