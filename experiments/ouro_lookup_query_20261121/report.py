"""Verify the frozen table cohort, routing statistic, and complete integer evidence."""
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
from query_deck import KEYS, PARENT, PREFIX, VARIANTS, build, eligible, history, summary, text
from query_stats import CONTROLS, LATTICE, MODES, summarize
from query_test import schema
from rule import MAX_NEW, parse

R=Path("results")


def canonical(x):
    return json.loads(json.dumps(x))


def check(x):
    assert x["tokens"] and len(x["tokens"])<=MAX_NEW and parse(x["text"])==x["number"]


def verify():
    manifest=load(R/"DECK.json")
    assert manifest==canonical(build()) and manifest["prior_sha256"]==sha(PARENT)
    n=load(R/"NATIVE.json")
    assert n["deck_sha256"]==sha(R/"DECK.json") and manifest["prefix"]==PREFIX
    by={}
    previous=history()
    targets=set()
    for r,t in zip(manifest["rows"],n["rows"],strict=True):
        assert r=={k:v for k,v in t.items() if k!="native"}
        assert r["id"] not in by and r["target"] not in targets
        by[r["id"]]=t
        targets.add(r["target"])
        a,b,c=[r["example"][x] for x in VARIANTS]
        assert a["order"]==b["order"] and c["order"].index(r["key"])!=a["order"].index(r["key"])
        assert b["values"]==c["values"] and sorted(a["values"])==sorted(b["values"])
        assert a["values"][r["key"]-1]==r["target"] and c["values"][r["key"]-1]==r["new"]!=r["target"]
        assert sum(x!=y for x,y in zip(a["values"],b["values"],strict=True))==2
        for k,e in r["example"].items():
            assert sorted(e["order"])==list(KEYS) and len(set(e["values"]))==4
            prompt=text(r["key"],**e)
            assert prompt not in previous
            previous.add(prompt)
            for p in t["native"][k]:
                check(p)
    assert len(by)==384
    s=load(R/"NATIVE_SUMMARY.json")
    assert canonical(summary(n["rows"]))==s
    v=dict(status="PASS",model_revision=REV,expected_weights_sha256=MODEL_HASH,
           weights_sha256=os.environ["OURO_ACTUAL_SHA256"],gpu=os.environ["OURO_GPU"],
           torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
           native_sha256=sha(R/"NATIVE.json"),deck_sha256=sha(R/"DECK.json"),scored_prompts=len(by)*3)
    assert v["weights_sha256"]==MODEL_HASH
    if not s["gate"]:
        return v
    pick=load(R/"QUERY_SITE.json")
    assert pick["native_sha256"]==v["native_sha256"] and pick["layers"]==24
    dev=pick["development"]
    assert [r["id"] for r in dev]==[r["id"] for r in n["rows"] if r["split"]=="development" and eligible(r,"original")]
    gain=[sum(x["gap"][1][l]-x["gap"][0][l] for x in dev)/len(dev) for l in range(24)]
    assert gain==pick["mean_routing_gain"] and pick["layer"]==max(range(24),key=lambda l:gain[l])
    cohort=load(R/"COHORT.json")
    raw=load(R/"QUERY_RAW.json")
    assert cohort["native_sha256"]==raw["native_sha256"]==v["native_sha256"]
    assert cohort["site_sha256"]==sha(R/"QUERY_SITE.json") and raw["cohort_sha256"]==sha(R/"COHORT.json")
    want=[dict(id=r["id"],test=schema(r)) for r in n["rows"] if r["split"]=="confirmation" and schema(r)]
    assert cohort["rows"]==want and [r["id"] for r in raw["rows"]]==[j["id"] for j in want]
    diag=sorted((r["id"] for r in want if "rescue" in r["test"]),key=lambda s:short_hash("q-causal",s))[:24]
    transfer=sorted((r["id"] for r in want if "crossed" in r["test"]),key=lambda s:short_hash("q-transfer",s))[:24]
    assert cohort["diagnostic_both"]==sorted(diag) and cohort["diagnostic_transfer"]==sorted(transfer)
    for r,w in zip(raw["rows"],want,strict=True):
        t=by[w["id"]]
        assert all(r[k]==t[k] for k in ("id","target","new","key","native","example"))
        assert r["test"]==w["test"]==list(r["pred"])
        for k,record in r["pred"].items():
            variant,phase=LATTICE[k]
            assert set(record)==set(MODES) and record["native"]==t["native"][variant][phase]
            for p in record.values():
                check(p)
        lo=set()
        if r["id"] in diag:
            lo.update(("rescue","reversal"))
        if r["id"] in transfer:
            lo.add("crossed")
        assert set(r["routing"])==lo
        for e in r["routing"].values():
            assert set(e)=={*MODES,"identity"}
            assert all(len(vals)==4 and all(0<=p<=1 for p in vals) and sum(vals)<=1+1e-6 for vals in e.values())
    assert canonical(summarize(raw["rows"]))==load(R/"QUERY_SUMMARY.json")
    v.update(raw_sha256=sha(R/"QUERY_RAW.json"),site_sha256=cohort["site_sha256"],
             cohort_sha256=raw["cohort_sha256"],runtime_s=raw["elapsed"],peak_allocated_mib=raw["peak_allocated_mib"])
    return v


def interval(x):
    lo,hi=x["interval"]
    if lo is None:
        return "无样本"
    return f"{x['total']:+d} 道，{x['rate']*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}] 个百分点"


def image(z):
    arms=("authentic","geometry","location")
    labels=("Authentic Q","Matched geometry","Location control")
    names=("K1 rescue","K2 old value","Dual success","Values only","Values and order")
    checks=("rescue","reversal","dual","values","crossed")
    fig,ax=plt.subplots(1,2,figsize=(10.8,3.65),gridspec_kw={"width_ratios":[1.35,1]})
    colors=("#4F728F","#9A5347","#ADB6BF")
    for i,(a,title) in enumerate(zip(arms,labels,strict=True)):
        vals=[z[k]["arms"][a] if k=="dual" else z[k]["arms"][a]["success"] for k in checks]
        n=[z[k]["n"] for k in checks]
        bars=ax[0].bar([j+(i-1)*.24 for j in range(5)],[100*v/max(t,1) for v,t in zip(vals,n,strict=True)],
                       width=.23,color=colors[i],label=title)
        ax[0].bar_label(bars,labels=[str(v) for v in vals],padding=1,fontsize=8)
    ax[0].set_xticks(range(5),names,rotation=16,ha="right")
    ax[0].set_ylabel("Exact target success (%)")
    ax[0].set_title("Free-generation outcomes",loc="left")
    ax[0].legend(frameon=False,fontsize=8)
    routes=("rescue","reversal","crossed")
    width=.245
    for i,(a,title) in enumerate(zip(("native","authentic","geometry"),("Native","Authentic Q","Matched geometry"),strict=True)):
        vals=[z["routing"][k]["mass"].get(a,{}).get("gap",0) for k in routes]
        ax[1].bar([j+(i-1)*width for j in range(3)],vals,width=.23,color=colors[i],label=title)
    ax[1].set_xticks(range(3),["K1 rescue","K2 old value","Crossed"])
    ax[1].set_ylabel("Correct minus original-wrong attention")
    ax[1].set_title("Locked layer diagnostic",loc="left")
    ax[1].legend(frameon=False,fontsize=8)
    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(axis="y",alpha=.15)
        a.set_axisbelow(True)
    fig.tight_layout()
    path=R/"LOOKUP_Q.png"
    fig.savefig(path,dpi=170,bbox_inches="tight")
    plt.close(fig)
    with Image.open(path) as f:
        assert f.width>1000 and f.height>420
        f.verify()
    return sha(path)


def render(v):
    n=load(R/"NATIVE_SUMMARY.json")
    d=["# 查表任务的单层输出查询 Q 双向干预","",
       "冻结 Ouro-1.4B Base。保留受体原生的 K 和 V，在一个物理层的当前输出位置替换全部查询头的 Q。每个输出 token 重新执行干预，判分使用完整整数。键值重排保留同一组整数，并将原答案移到另一键。","",
       "| 320 道确认题 | K1 正答 | K2 正答 | 错转对 | 其他表值转正答 | 对转错 |",
       "|---|---:|---:|---:|---:|---:|"]
    for k,s in n["confirmation"].items():
        d.append(f"| {k} | {s['k1']} | {s['k2']} | {s['corrected']} | {s['wrong_table_to_correct']} | {s['regressed']} |")
    if not n["gate"]:
        d.extend(["","原生纠正题数未通过预设门槛。"])
    else:
        pick=load(R/"QUERY_SITE.json")
        layer=pick["layer"]
        s=load(R/"QUERY_SUMMARY.json")
        d.extend(["",f"开发集仅按注意力差锁定物理第 {layer+1} 层。依据为 {len(pick['development'])} 道从其他表项值转正答的题，正确表项减原错表项的平均注意力变化为 {pick['mean_routing_gain'][layer]:+.4f}。","",
                  "| 终点 | 道数 | 真 Q 命中 | 等角等范数 | 分号位置对照 | 真 Q 命中原供体值 |",
                  "|---|---:|---:|---:|---:|---:|"])
        for k in ("rescue","reversal","dual","values","crossed"):
            z=s[k]
            arms=z["arms"]
            if k=="dual":
                d.append(f"| 同题同时救回及回退 | {z['n']} | {arms['authentic']} | {arms['geometry']} | {arms['location']} | — |")
            else:
                c="—" if k in ("rescue","reversal") else arms["authentic"]["source_correct"]
                label={"rescue":"K1 恢复正确值","reversal":"K2 重现原错","values":"只换值","crossed":"换值并换行"}[k]
                d.append(f"| {label} | {z['n']} | {arms['authentic']['success']} | {arms['geometry']['success']} | {arms['location']['success']} | {c} |")
        d.extend(["",f"迁移队列与原题自然纠正队列重叠 {s['values']['overlap_original']}（只换值）和 {s['crossed']['overlap_original']}（换值并换行）道。迁移按新表实际 K1 错、K2 对且源表 K2 正确判定，因此规模可以不同。","",
                  "## 对照结果",""])
        for k in ("rescue","reversal","dual","values","crossed"):
            for a,x in s[k]["paired"].items():
                d.append(f"- {k} 对 {a} 的配对 97.5% 区间为 {interval(x)}。")
        for a,x in s["dual"]["bounds"].items():
            d.extend(["",f"真实 Q 相比 {a} 的额外双向成功为 {x['events']}/{x['n']} 个答案簇，单侧 97.5% 上界 {x['upper_97_5']*100:.2f}%。"])
        for k,f in s["gates"].items():
            d.extend(["",f"{k} {'通过' if f else '未通过'}。"])
        d.extend(["","诊断图按固定子集记录正确表项和原错表项的注意力质量差。该指标只作路由诊断，整答和双向结果分别检验。分号对照匹配位移范数及相对受体 Q 的径向比例，位置不同可拥有不同原始 Q 范数。","",
                  "![完整生成和表项注意力](results/LOOKUP_Q.png)","",
                  "本轮约束锁定层的输出查询和该查表协议。原生 K1→K2 增益还可来自表项键值、较早的查询形成、其他层或多处共同计算。"])
        v["figure_sha256"]=image(s)
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__=="__main__":
    v=verify()
    render(v)
    save(R/"VERIFICATION.json",v)
    print(v,flush=True)
