"""Verify both lookup domains, the fixed-suffix identity, and onset decisions."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import transformers

from edge import short_hash
from native import MODEL_HASH, REV, load, save, sha
from onset_deck import DOMAINS, PARENT, VARIANTS, build, history, pair, prompt, repair, summary, supplement
from onset_stats import MODES, SITES, summarize
from rule import MAX_NEW, parse

R=Path("results")


def equal(a,b):
    assert json.loads(json.dumps(a))==b


def integer(x):
    assert 0<len(x["tokens"])<=MAX_NEW and parse(x["text"])==x["number"]


def verify_native():
    d=load(R/"DECK.json")
    n=load(R/"NATIVE.json")
    p=load(R/"PAIRED.json")
    equal(build(),d)
    assert d["parent_sha256"]==sha(PARENT)
    assert n["deck_sha256"]==sha(R/"DECK.json") and p["native_sha256"]==sha(R/"NATIVE.json")
    seen=history()
    for a,b in zip(d["rows"],n["rows"],strict=True):
        assert a=={k:v for k,v in b.items() if k!="native"}
        s=prompt(a,"original")
        assert s not in seen
        seen.add(s)
        for w in b["native"]["original"]:
            integer(w)
    chosen=[r for r in n["rows"] if repair(r)]
    ids={}
    for r,q in zip(chosen,p["rows"],strict=True):
        z=supplement(r)
        assert z==dict(q,native={"original":q["native"]["original"]})
        ids[q["id"]]=q
        for k in VARIANTS[1:]:
            s=prompt(q,k)
            assert s not in seen
            seen.add(s)
            for w in q["native"][k]:
                integer(w)
    native=load(R/"NATIVE_SUMMARY.json")
    equal(summary(n["rows"],p["rows"]),native)
    v=dict(status="PASS",model_revision=REV,expected_weights_sha256=MODEL_HASH,
           weights_sha256=os.environ["OURO_ACTUAL_SHA256"],gpu=os.environ["OURO_GPU"],
           torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
           deck_sha256=sha(R/"DECK.json"),native_sha256=sha(R/"NATIVE.json"),
           paired_sha256=sha(R/"PAIRED.json"),prompts=len(n["rows"])+3*len(p["rows"]))
    assert MODEL_HASH==v["weights_sha256"]
    return native,v,ids,p


def verify():
    native,v,ids,p=verify_native()
    if not native["gate"]:
        return v
    c=load(R/"COHORT.json")
    e=load(R/"ONSET_RAW.json")
    assert c["paired_sha256"]==e["paired_sha256"]==v["paired_sha256"]
    assert e["cohort_sha256"]==sha(R/"COHORT.json")
    assert c["phase"]==[r["id"] for r in p["rows"]]==[r["id"] for r in e["rows"]]
    assert c["content"]==[r["id"] for r in p["rows"] if pair(r)]
    ref={k:sorted((r["id"] for r in p["rows"] if r["domain"]==k and pair(r)),
                  key=lambda x:short_hash("onset replay",x))[:8] for k in DOMAINS}
    assert c["audit"]==ref
    fixed={i for xs in ref.values() for i in xs}
    for r in e["rows"]:
        src=ids[r["id"]]
        assert all(r[k]==src[k] for k in ("domain","target","old","native","example"))
        assert r["permitted"]==pair(src)
        assert set(r["pred"])==set(map(str,SITES))
        for layer,g in r["pred"].items():
            assert set(g)==({"phase","content"} if pair(src) else {"phase"})
            for k,directions in g.items():
                assert set(directions)=={"forward","reverse"}
                for arms in directions.values():
                    assert set(arms)==set(MODES[k])
                    for ans in arms.values():
                        integer(ans)
        assert ("parity" in r)==(r["id"] in fixed)
        if "parity" in r:
            assert r["parity"]["n"]==20 and 0<=r["parity"]["max_abs"]<.01
    equal(summarize(e["rows"]),load(R/"ONSET_SUMMARY.json"))
    v.update(raw_sha256=sha(R/"ONSET_RAW.json"),cohort_sha256=e["cohort_sha256"],
             runtime_s=e["elapsed"],peak_allocated_mib=e["peak_allocated_mib"])
    return v


def label(x):
    return "入轮" if x==1 else f"第 {x} 层入口"


def interval(z):
    lo,hi=z["simultaneous"]
    return f"{100*z['mean']:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]" if lo is not None else "无样本"


def plot(z):
    fig,axes=plt.subplots(1,2,figsize=(9.7,3.8))
    colors={"standard":"#3A688A","shifted":"#A55243"}
    for ax,k,name in zip(axes,MODES,("Same-table phase exchange","Paired-table value exchange"),strict=True):
        for d in DOMAINS:
            for a in ("authentic","geometry"):
                x=[z["layers"][str(l)][d][k] for l in SITES]
                ax.plot(range(5),[100*t["counts"][a]["dual"]/max(1,t["n"]) for t in x],
                        color=colors[d],linestyle="-" if a=="authentic" else "--",marker="o" if a=="authentic" else "s",
                        markersize=4,label=f"{d} {'state swap' if a=='authentic' else 'geometry'}")
        ax.set_xticks(range(5),["Entry","L6","L12","L18","L20"])
        ax.set_title(name,loc="left")
        ax.set_ylabel("Exact dual success (%)")
        ax.legend(frameon=False,fontsize=7)
        ax.set_ylim(0,max(3,ax.get_ylim()[1]*1.12))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
    fig.tight_layout()
    dst=R/"LOOKUP_ONSET.png"
    fig.savefig(dst,dpi=170,bbox_inches="tight")
    plt.close(fig)
    with Image.open(dst) as i:
        assert i.width>1000 and i.height>430
        i.verify()
    return sha(dst)


def render(v):
    n=load(R/"NATIVE_SUMMARY.json")
    d=["# 冻结后续写入时，查表值在何处变得可用","",
       "使用冻结的 Ouro-1.4B Base 与自然完整整数生成。五个位置预先固定。分别交换同题两轮的残差和同轮两表的残差，之后每一层保留受体本来的注意力、MLP 写入。用改变当前末 token 的残差差值与原生 RMSNorm、LM 头得到下一 token，并用物理回放核对。第二分布同时改变键名和表格样式。","",
       "| 分布 | 原生 K1 正答 | 原生 K2 正答 | 精确自然纠正 | 合格配对表 |",
       "|---|---:|---:|---:|---:|"]
    for x in DOMAINS:
        p=n[x]
        d.append(f"| {x} | {p['native_k1']}/{p['n']} | {p['native_k2']}/{p['n']} | {p['exact']} | {p['pairs']} |")
    if not n["gate"]:
        d.extend(["","原生队列规模未通过预设的跨分布门槛。"])
    else:
        z=load(R/"ONSET_SUMMARY.json")
        for kind in MODES:
            title="同题跨轮" if kind=="phase" else "同轮跨表"
            d.extend(["",f"## {title}的完整整数","",
                      "| 层界 | 分布 | n | 真状态正向 / 反向 / 双向 | 几何双向 | 无关双向 |",
                      "|---|---|---:|---:|---:|---:|"])
            for l in SITES:
                for name in DOMAINS:
                    a=z["layers"][str(l)][name][kind]
                    v0=a["counts"]["authentic"]
                    r=a["counts"]["unrelated"]["dual"] if kind=="content" else "—"
                    d.append(f"| {label(l)} | {name} | {a['n']} | {v0['forward']} / {v0['reverse']} / {v0['dual']} | {a['counts']['geometry']['dual']} | {r} |")
        d.extend(["","## 预设判定","",
                  "下表每格为真实双向命中减匹配对照的百分点差及同时区间。区间按 30 项 Bonferroni 家族校正。"])
        for l in SITES:
            for q in DOMAINS:
                b=z["layers"][str(l)][q]
                d.extend(["",f"- {label(l)}，{q}。跨轮对几何 {interval(b['phase']['contrasts']['geometry'])}。"
                          f"跨表对几何 {interval(b['content']['contrasts']['geometry'])}，对无关值 {interval(b['content']['contrasts']['unrelated'])}。"])
        s=z["decision"]
        first=label(s["first"]) if s["first"] is not None else "未达到联合门槛"
        d.extend(["",f"第 20 层配对参照{'通过' if s['reference'] else '未通过'}。最早通过跨轮与跨表的边界为 {first}。"])
        if prev:=s["preceding"]:
            q=s["preceding_maximum"]
            d.extend(["",f"前一测试点为 {label(prev)}。其两个分布与各对照的单侧 97.5% 最大上界 {q['upper_97_5']*100:.2f}%，跨五层保守最大上界 {q['upper_across_sites']*100:.2f}%。"
                      f"预设的联合效应区间判据{'通过' if s['identified_interval'] else '未通过'}。"])
        early=z["layers"]["1"]["shifted"]["content"]
        if early["supported"]:
            c=early["counts"]
            d.extend(["",f"新格式在入轮时已达到同轮跨表判据。真状态双向 {c['authentic']['dual']}/{early['n']}，"
                      f"几何对照 {c['geometry']['dual']}/{early['n']}，无关行对照 {c['unrelated']['dual']}/{early['n']}。"
                      "因此区间只定位跨轮纠错与跨表值转移同时通过的边界。单独的跨表信号随层界非单调，且可能依赖锁定的后续写入。"])
        diag=z["diagnostic"]
        d.extend(["",f"固定 {diag['records']} 个题对执行 {diag['checks']} 次实际物理回放，末端残差和加法预测的最大绝对误差为 {diag['max_abs']:.3g}。"
                  "结论取决于这组固定读取条件和实际生成；未测试的中间层以及其他读取方式仍可能不同。","",
                  "![五个边界的双向整答](results/LOOKUP_ONSET.png)"])
        v["figure_sha256"]=plot(z)
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__=="__main__":
    v=verify()
    render(v)
    save(R/"VERIFICATION.json",v)
    print(v,flush=True)
