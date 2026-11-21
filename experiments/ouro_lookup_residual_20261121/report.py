"""Audit actual lookup swaps and render integer and residual results."""
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
from origin_stats import paired
from query_deck import PREFIX, text
from residual_deck import PARENT, VARIANTS, augment, build, history, is_pair, natural_repair, summary
from residual_stats import ARMS, CONTROLS, SIDES, exact, summarize
from rule import MAX_NEW, parse

R=Path("results")


def same(a,b):
    assert json.loads(json.dumps(a))==b


def check(q):
    assert 0<len(q["tokens"])<=MAX_NEW and parse(q["text"])==q["number"]


def verify():
    deck=load(R/"DECK.json")
    a=load(R/"NATIVE.json")
    b=load(R/"PAIR_NATIVE.json")
    assert deck["prior_sha256"]==sha(PARENT) and deck["prefix"]==PREFIX
    same(build(),deck)
    assert a["deck_sha256"]==sha(R/"DECK.json") and b["native_sha256"]==sha(R/"NATIVE.json")
    previous=history()
    for x,y in zip(deck["rows"],a["rows"],strict=True):
        assert x=={k:v for k,v in y.items() if k!="native"}
        q=text(x["key"],**x["example"]["original"])
        assert q not in previous
        previous.add(q)
        for z in y["native"]["original"]:
            check(z)
    base=[x for x in a["rows"] if natural_repair(x)]
    pairs=b["rows"]
    by={}
    for x,y in zip(base,pairs,strict=True):
        z=augment(x)
        assert {k:v for k,v in z.items() if k!="native"}=={k:v for k,v in y.items() if k!="native"}
        assert y["native"]["original"]==x["native"]["original"]
        assert y["example"]["swapped"]["values"][y["key"]-1]==y["old"]
        assert y["example"]["unrelated_original"]["values"][y["key"]-1]==y["target"]
        assert y["example"]["unrelated_swapped"]["values"][y["key"]-1]==y["old"]
        by[y["id"]]=y
        for tag in VARIANTS[1:]:
            q=text(y["key"],**y["example"][tag])
            assert q not in previous
            previous.add(q)
            for w in y["native"][tag]:
                check(w)
    n=load(R/"NATIVE_SUMMARY.json")
    same(summary(a["rows"],pairs),n)
    v=dict(status="PASS",model_revision=REV,expected_weights_sha256=MODEL_HASH,
           weights_sha256=os.environ["OURO_ACTUAL_SHA256"],gpu=os.environ["OURO_GPU"],
           torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
           deck_sha256=sha(R/"DECK.json"),native_sha256=sha(R/"NATIVE.json"),
           pair_native_sha256=sha(R/"PAIR_NATIVE.json"),prompts=len(a["rows"])+3*len(pairs))
    assert v["weights_sha256"]==MODEL_HASH
    if not n["gate"]:
        return v
    c=load(R/"COHORT.json")
    r=load(R/"RESIDUAL_RAW.json")
    assert c["physical_layer"]==20 and c["pair_native_sha256"]==r["pair_native_sha256"]==v["pair_native_sha256"]
    assert r["cohort_sha256"]==sha(R/"COHORT.json")
    keep=[x["id"] for x in pairs if is_pair(x)]
    assert c["eligible"]==keep==[x["id"] for x in r["rows"]]
    probes=sorted(keep,key=lambda x:short_hash("residual frozen diagnostic",x))[:24]
    assert sorted(probes)==c["diagnostic"]
    for x in r["rows"]:
        source=by[x["id"]]
        assert all(x[k]==source[k] for k in ("id","target","old","key","native","example"))
        assert set(x["pred"])==set(SIDES)
        for s in SIDES:
            assert set(x["pred"][s])==set(ARMS) and x["pred"][s]["native"]==source["native"][s][1]
            for z in x["pred"][s].values():
                check(z)
        assert ("diagnostics" in x)==(x["id"] in probes)
        for k,b in x.get("diagnostics",{}).items():
            assert k in SIDES and set(b)=={"swap_locked","geometry_locked","irrelevant_locked"}
            for z in b.values():
                assert z["delta_norm"]>0 and 0<=z["error"] and 0<=z["k_v_max"] and 0<=z["layer20_writes_max"]
    same(summarize(r["rows"]),load(R/"RESIDUAL_SUMMARY.json"))
    v.update(raw_sha256=sha(R/"RESIDUAL_RAW.json"),cohort_sha256=r["cohort_sha256"],
             runtime_s=r["elapsed"],peak_allocated_mib=r["peak_allocated_mib"])
    return v


def line(x):
    l,h=x["interval"]
    if l is None:
        return "无样本"
    return f"{x['total']:+d} 次，{x['rate']*100:+.1f} [{l*100:+.1f}, {h*100:+.1f}] 个百分点"


def by_direction(rows):
    def recovery_delta(row,side,kind):
        return int(exact(row,side,kind+"_adaptive"))-int(exact(row,side,kind+"_locked"))
    return {s:{control:paired(rows,[recovery_delta(r,s,"swap")-recovery_delta(r,s,control) for r in rows])
               for control in CONTROLS} for s in SIDES}


def plot(z):
    fig,axes=plt.subplots(1,2,figsize=(9.6,3.6))
    choices=("own","swap","geometry","irrelevant")
    labels=("Own bypass","Swapped","Geometry","Unrelated")
    cols=("#467194","#A05241")
    for q,ref in enumerate(("adaptive","locked")):
        a=[sum(z["outcome"][s][k+"_"+ref]["receiving"] for s in SIDES) for k in choices]
        b=axes[0].bar([i+(q-.5)*.32 for i in range(4)],a,width=.3,color=cols[q],
                     label=("Adaptive writes","Receiver-fixed writes")[q])
        axes[0].bar_label(b,padding=1,fontsize=8)
    axes[0].set_xticks(range(4),labels,rotation=12)
    axes[0].set_ylabel("Exact recipient answers (both directions)")
    axes[0].set_title("Bypass and downstream recomputation",loc="left")
    axes[0].legend(frameon=False,fontsize=8)
    for j,ref in enumerate(("locked","fixed_other")):
        a=[z["dual_other"][k+"_"+ref] for k in ("own","swap")]
        b=axes[1].bar([i+(j-.5)*.32 for i in range(2)],a,width=.3,color=cols[j],
                     label=("Receiver-fixed writes","Opposite-table fixed writes")[j])
        axes[1].bar_label(b,padding=1,fontsize=8)
    axes[1].set_xticks(range(2),["Own bypass","Swapped bypass"])
    axes[1].set_ylabel("Pairs switching both answers")
    axes[1].set_title("Fixed-write reference",loc="left")
    axes[1].legend(frameon=False,fontsize=8)
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(axis="y",alpha=.17)
        a.set_ylim(0,max(1,a.get_ylim()[1]*1.15))
        a.set_axisbelow(True)
    fig.tight_layout()
    p=R/"RESIDUAL_BYPASS.png"
    fig.savefig(p,dpi=170,bbox_inches="tight")
    plt.close(fig)
    with Image.open(p) as g:
        assert g.width>1000 and g.height>420
        g.verify()
    return sha(p)


def render(v):
    s=load(R/"NATIVE_SUMMARY.json")
    r,p=s["original"],s["paired"]
    d=["# 从第 20 层残差旁路和后续写入追踪查表答案","",
       "在冻结的 Ouro-1.4B Base 中，先把 K2 输出位置的第 20 层 Q 换为同一输入的 K1 Q。在该层完成真实注意力和 MLP 写入后，只加上另一表的第 20 层原始输入与受体原始输入之差，从而单独替换残差旁路。后续四层按条件重算，或回放指定表的原生 A/MLP 写入。","",
       f"384 道原题的原生 K1 / K2 正答各为 {r['k1']} 和 {r['k2']}，{r['wrong_table']} 道由错误表值转为正答。符合四组 K2 均正确条件的独立表对为 {p['eligible']}。"]
    if s["gate"]:
        e=load(R/"RESIDUAL_SUMMARY.json")
        d.extend(["","| 干预 | 原表正确 | 原表跟随配对值 | 配对表正确 | 配对表跟随原值 | 同对双向转移 |",
                  "|---|---:|---:|---:|---:|---:|"])
        for a in ARMS:
            x,y=[e["outcome"][k][a] for k in SIDES]
            d.append(f"| {a} | {x['receiving']} | {x['other_table']} | {y['receiving']} | {y['other_table']} | {e['dual_other'][a]} |")
        d.extend(["","## 预设机制判定","",
                  f"在真实旁路互换条件下，允许后续层重算比固定写入多救回 {e['compensation_counts']['helped']} 个受体方向，也有 {e['compensation_counts']['harmed']} 个方向反转为错误。"])
        for k,q in e["contrasts"].items():
            d.extend(["",f"- {k} 对照下，双向转移差 {line(q['residual'])}，后层重算特异效应 {line(q['compensation'])}。"])
        for k,q in e["upper_bounds"].items():
            d.extend(["",f"- {k} 的残差转移与后层补偿额外事件率单侧 97.5% 上界分别为 "
                      f"{100*q['residual']['upper_97_5']:.2f}%、{100*q['compensation']['upper_97_5']:.2f}%。"])
        for k,q in e["gates"].items():
            d.extend(["",f"{k} {'通过' if q else '未通过'}。"])
        raw=load(R/"RESIDUAL_RAW.json")["rows"]
        already=sum(q["native"]["swapped"][0]["number"]==q["old"] for q in raw)
        d.extend(["","## 事后按受体方向分层","",
                  f"配对表有 {already}/{len(raw)} 道在 K1 已答对。以下分层只描述合并效应来源，保留预设的合并判定。"])
        for key,z in by_direction(raw).items():
            label="原表自然纠正方向" if key=="original" else "配对表方向"
            d.extend(["",f"{label}相对几何、无关换值对照的重算效应，分别为 {line(z['geometry'])}、{line(z['irrelevant'])}。"])
        d.extend(["","## 回放参考与代数校验",""])
        for k,q in e["different_reference"].items():
            d.append(f"- 对 {k} 旁路，改用另一表的固定写入时，输出转向另一表值的配对变化为 {line(q)}。")
        diag=e["diagnostic"]
        d.extend(["",f"固定 {diag['pairs']} 对诊断题的第 20 层 K/V、真实分支写入及最后一层残差均通过检查。固定后续写入时，末层归一化前的实测差与注入旁路差的最大绝对偏差为 {diag['max_abs']['error']:.3g}。所有干预对每个实际生成 token 重算。","",
                  "![残差旁路与后层写入](results/RESIDUAL_BYPASS.png)","",
                  "固定写入来自某个正常计算参考，参考状态可携带表值。两种参考都保留在表中。剩余推断限于该确认集、第 20 层答前位置和这类配对交换。"])
        v["figure_sha256"]=plot(e)
    else:
        d.extend(["","原生确认门槛未通过。"])
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__=="__main__":
    v=verify()
    render(v)
    save(R/"VERIFICATION.json",v)
    print(v,flush=True)
