"""Verify frozen cohorts and exact-answer effects, then render the factorial report."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import transformers

from branches_deck import PARENT, build, history, paired_variant, summary, swapped_repair
from branches_stats import ARMS, PATTERNS, SCOPES, summarize
from edge import short_hash
from native import MODEL_HASH, REV, load, save, sha
from query_deck import text
from residual_deck import natural_repair
from rule import MAX_NEW, parse

R=Path("results")


def equal(x,y):
    assert json.loads(json.dumps(x))==y


def integer(p):
    assert 0<len(p["tokens"])<=MAX_NEW and parse(p["text"])==p["number"]


def group(r):
    n=r["native"]["swapped"][0]["number"]
    return "paired_correct_k1" if n==r["old"] else ("paired_natural_repair" if swapped_repair(r) else "paired_other_k1_error")


def verify():
    d=load(R/"DECK.json")
    a=load(R/"NATIVE.json")
    b=load(R/"PAIR_NATIVE.json")
    assert d["parent_sha256"]==sha(PARENT/"PAIR_NATIVE.json")
    equal(build(),d)
    assert a["deck_sha256"]==sha(R/"DECK.json") and b["native_sha256"]==sha(R/"NATIVE.json")
    seen=history()
    for x,y in zip(d["rows"],a["rows"],strict=True):
        assert x=={k:v for k,v in y.items() if k!="native"}
        p=text(x["key"],**x["example"]["original"])
        assert p not in seen
        seen.add(p)
        for z in y["native"]["original"]:
            integer(z)
    want=[x for x in a["rows"] if natural_repair(x)]
    schedule=[]
    index={}
    for x,y in zip(want,b["rows"],strict=True):
        raw=paired_variant(x)
        assert raw==dict({k:u for k,u in y.items() if k!="native"},native={"original":y["native"]["original"]})
        p=text(y["key"],**y["example"]["swapped"])
        assert p not in seen
        seen.add(p)
        for z in y["native"]["swapped"]:
            integer(z)
        sid=y["id"]
        index[sid]=y
        schedule.append([sid,"original"])
        if y["native"]["swapped"][1]["number"]==y["old"]:
            schedule.append([sid,group(y)])
    s=load(R/"NATIVE_SUMMARY.json")
    equal(summary(a["rows"],b["rows"]),s)
    proof=dict(status="PASS",model_revision=REV,expected_weights_sha256=MODEL_HASH,
               weights_sha256=os.environ["OURO_ACTUAL_SHA256"],gpu=os.environ["OURO_GPU"],
               torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
               native_prompts=len(a["rows"])+len(b["rows"]),deck_sha256=sha(R/"DECK.json"),
               native_sha256=sha(R/"NATIVE.json"),pair_sha256=sha(R/"PAIR_NATIVE.json"))
    assert proof["weights_sha256"]==MODEL_HASH
    if not s["gate"]:
        return proof
    lock=load(R/"COHORT.json")
    raw=load(R/"BRANCH_RAW.json")
    assert lock["outputs"]==schedule==[[r["id"],r["group"]] for r in raw["rows"]]
    assert raw["pair_native_sha256"]==lock["pair_native_sha256"]==proof["pair_sha256"]
    assert raw["cohort_sha256"]==sha(R/"COHORT.json")
    selected=sorted(index,key=lambda k:short_hash("branch replay",k))[:24]
    assert sorted(selected)==lock["diagnostics"]
    for row in raw["rows"]:
        r=index[row["id"]]
        tag="original" if row["group"]=="original" else "swapped"
        assert row["source_target"]==r["target"]
        assert row["side"]==tag and row["target"]==r["target" if tag=="original" else "old"]
        assert row["native"]==r["native"][tag] and row["foil"]==r["native"][tag][0]["number"]
        assert set(row["pred"])==set(ARMS) and row["pred"]["native"]==row["native"][1]
        for z in row["pred"].values():
            integer(z)
        assert ("additive_error" in row)==(tag=="original" and row["id"] in selected)
        assert all(0<=v<.01 for v in row.get("additive_error",{}).values())
    equal(summarize(raw["rows"]),load(R/"BRANCH_SUMMARY.json"))
    proof.update(raw_sha256=sha(R/"BRANCH_RAW.json"),cohort_sha256=raw["cohort_sha256"],
                 runtime_s=raw["elapsed"],peak_allocated_mib=raw["peak_allocated_mib"])
    return proof


def interval(p):
    lo,hi=p["interval"]
    return "无样本" if lo is None else f"{p['total']:+d} 道，{p['rate']*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}] 个百分点"


def image(z):
    fig,axes=plt.subplots(1,2,figsize=(9.8,3.7))
    labels=("Attention only","MLP only","Both branches")
    groups=("attention","mlp","both")
    original=z["groups"]["original"]
    colors=("#3F6D8E","#A65851")
    for ax,scope,title in zip(axes,SCOPES,("Output position","Entire prefix"),strict=True):
        for j,tag in enumerate(("authentic","geometry")):
            counts=[original["arms"][f"{scope}_{tag}_{k}"]["wrong_k1"] for k in groups]
            a=ax.bar([i+(j-.5)*.33 for i in range(3)],counts,width=.31,color=colors[j],
                     label=("K1 writes","Matched geometry")[j])
            ax.bar_label(a,padding=2,fontsize=8)
        ax.set_xticks(range(3),labels)
        ax.set_ylabel("Exact returns to the K1 error")
        ax.set_ylim(0,max(3,ax.get_ylim()[1]*1.18))
        ax.set_title(title,loc="left")
        ax.legend(frameon=False,fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y",alpha=.16)
        ax.set_axisbelow(True)
    fig.tight_layout()
    dst=R/"LATE_BRANCH_REPLAY.png"
    fig.savefig(dst,dpi=170,bbox_inches="tight")
    plt.close(fig)
    with Image.open(dst) as f:
        assert f.width>1100 and f.height>430
        f.verify()
    return sha(dst)


def render(v):
    base=load(R/"NATIVE_SUMMARY.json")
    d=["# 自然查表纠错中的后四层注意力和 MLP 写入","",
       "冻结 Ouro-1.4B Base。在当前生成前缀的原生 K2 状态上，分别回放同题 K1 的第 21–24 层注意力与 MLP 写入。输出位置和整个有效前缀作为两种范围单独检验。完整生成需重现原题的另一表项值或完整正答。","",
       f"原表 {base['n']} 道，原生 K1 / K2 正答分别 {base['original_k1']} / {base['original_k2']} 道，原错为另一表项且 K2 正确的主队列为 {base['original_exact']} 道。配对表 K2 正答 {base['swapped_k2']} 道，其中同样从错误表项转正答 {base['swapped_exact']} 道，K1 已正确 {base['swapped_correct_both']} 道。"]
    if not base["gate"]:
        d.extend(["","原生队列规模未通过预设门槛。"])
    else:
        z=load(R/"BRANCH_SUMMARY.json")
        for name,g in z["groups"].items():
            if not g["n"]:
                continue
            d.extend(["",f"## {name}，{g['n']} 道","",
                      "| 回放范围 | 写入来源 | 注意力原错 / 正答 | MLP 原错 / 正答 | 双回放原错 / 正答 |",
                      "|---|---|---:|---:|---:|"])
            for scope in SCOPES:
                for q in ("authentic","geometry"):
                    a=[g["arms"][f"{scope}_{q}_{p}"] for p in ("attention","mlp","both")]
                    d.append(f"| {scope} | {q} | "+" | ".join(f"{r['wrong_k1']} / {r['correct']}" for r in a)+" |")
            for scope,s in g["scopes"].items():
                v1,v2=[s["totals"][k] for k in ("authentic","geometry")]
                d.extend(["",f"{scope} 的注意力恢复、MLP 恢复和两者任一恢复即足够的计数分别为 "
                          f"{v1['attention_restore']}、{v1['mlp_restore']}、{v1['redundancy']} 道，"
                          f"几何为 {v2['attention_restore']}、{v2['mlp_restore']}、{v2['redundancy']} 道。"])
                for k in PATTERNS:
                    q=s["bounds"][k]
                    d.extend(["",f"- {scope} {k} 对几何的配对 97.5% 区间是 {interval(s['paired'][k])}。"
                              f"额外成功的单侧 97.5% 上界为 {q['upper_97_5']*100:.2f}%，"
                              f"支持门槛{'通过' if s['gates'][k] else '未通过'}。"])
                d.extend(["",f"有符号整答交互相对几何为 {interval(s['paired']['signed_interaction'])}。"])
        h=z["diagnostic"]
        d.extend(["","## 物理回放核对","",
                  f"固定 {h['n']} 道原表诊断题验证无改动回放及末层预归一化残差的加法恒等式，最大绝对误差 {h['max_abs']:.3g}。"
                  f"原表两个空间范围的三项 10% 门槛{'均通过' if z['close_late_suffix'] else '未全部通过'}。"
                  "该判定只约束预设的整答模式及后四层回放。","",
                  "![完整生成中原错的复现次数](results/LATE_BRANCH_REPLAY.png)"])
        v["figure_sha256"]=image(z)
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__=="__main__":
    v=verify()
    render(v)
    save(R/"VERIFICATION.json",v)
    print(v,flush=True)
