"""Check the locked cohort, exact outputs and the reported route comparisons."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import transformers

from bidi import unordered
from borrow import needs
from edge import short_hash
from native import MODEL_HASH, REV, load, save, sha
from native_deck import is_repair, report as native_summary
from path_deck import PARENT, history
from path_stats import ARMS, PATHS, summarize
from rule import PREFIX, parse

R=Path("results")


def equals(x,y):
    assert json.loads(json.dumps(x))==y


def check(answer):
    assert answer["tokens"] and parse(answer["text"])==answer["number"]


def verify():
    ordinary=load(R/"DECK.json")
    ref=load(R/"NATIVE.json")
    used=history()
    grouped=ordinary["groups"]
    assert ordinary["prior_main_sha256"]==sha(PARENT/"DECK.json")
    assert ordinary["prior_extension_sha256"]==sha(PARENT/"extension"/"DECK.json")
    assert ordinary["earlier_unordered"]==len(used) and ordinary["prompt"]==PREFIX
    assert ref["deck_sha256"]==sha(R/"DECK.json")
    by={}
    for a,b in zip(ordinary["rows"],ref["rows"],strict=True):
        assert a=={k:v for k,v in b.items() if k!="pred"} and a["id"] not in by
        by[a["id"]]=b
        assert a["target"]+100==a["foil"] and grouped[str(a["target"])]==a["split"]
        for k in ("borrow","safe"):
            v=a[k]
            a0,b0=v[0]
            a1,b1=v[-1]
            assert v[1:3]==[[a1,b0],[a0,b1]] and [a1-a0,b0-b1]==a[k+"_steps"]
            assert all(len(str(x))==3 for q in v for x in q)
            assert a0//100==a1//100 and b0//100==b1//100 and b1-a1==a["target"]
            assert [needs(*q) for q in v]==[False,False,False,k=="borrow"]
            if k=="borrow":
                assert b0-a0==a["foil"]
            for q in v:
                u=unordered(q)
                assert u not in used
                used.add(u)
        assert len(b["pred"])==2 and all(len(g)==5 for g in b["pred"])
        for seq in b["pred"]:
            for p in seq:
                check(p)
    assert len(by)==340 and len(used)-ordinary["earlier_unordered"]==2720
    summary=load(R/"NATIVE_SUMMARY.json")
    equals(native_summary(ref["rows"],grouped),summary)
    v=dict(status="PASS",model_revision=REV,expected_weights_sha256=MODEL_HASH,
           weights_sha256=os.environ["OURO_ACTUAL_SHA256"],gpu=os.environ["OURO_GPU"],
           torch=torch.__version__, transformers=transformers.__version__,cuda=torch.version.cuda,
           native_sha256=sha(R/"NATIVE.json"),deck_sha256=sha(R/"DECK.json"),scored_prompts=1700)
    assert v["expected_weights_sha256"]==v["weights_sha256"]
    if not summary["gate"]:
        return v
    lock=load(R/"COHORT.json")
    output=load(R/"PATH_RAW.json")
    assert lock["natural_sha256"]==output["native_sha256"]==v["native_sha256"]
    assert output["cohort_sha256"]==sha(R/"COHORT.json")
    order=sorted(q["id"] for q in ref["rows"] if q["split"]=="confirmation" and is_repair(q))
    assert [r["id"] for r in lock["pairs"]]==order
    h=sorted(order,key=lambda i:short_hash("routes",i))
    assert lock["component_ids"]==sorted(h[:24]) and lock["deeper_ids"]==sorted(h[:48])
    plan={r["id"]:r for r in lock["pairs"]}
    actual=output["rows"]
    assert [(r["id"],r["kind"]) for r in actual]==[(i,k) for i in order for k in ("repair","safe")]
    for r in actual:
        p=plan[r["id"]]
        source=by[r["source_id"]]
        assert r["source_id"]==(r["id"] if r["kind"]=="repair" else p["safe"])
        assert [(u["target"],u["foil"]) for u in (r,source,by[r["id"]])]==[(r["target"],r["foil"])]*3
        ix=3 if r["kind"]=="repair" else 4
        assert r["native"]==[g[ix] for g in source["pred"]] and r["distance"]==p["difference"]
        assert set(r["pred"])==set(ARMS) and r["pred"]["native"]==r["pred"]["restore_joint"]==r["native"][1]
        for i in r["pred"].values():
            check(i)
        if r["aligned"]:
            c=r["query_counts"]
            assert c["pivot"]==1 and c["context"]>0 and c["context"]+1==c["joint"]
            for m in r["masses"].values():
                assert 0<=m["matched"]<=m["available"]+1e-5 and m["full"]>0
        else:
            assert all(a==r["native"][1] for a in r["pred"].values())
        assert ("components" in r)==(r["kind"]=="repair" and r["aligned"] and r["id"] in lock["component_ids"])
        assert ("depth" in r)==(r["kind"]=="repair" and r["aligned"] and r["id"] in lock["deeper_ids"])
        for d in r.get("depth",{}).values():
            assert set(d["pred"])=={"native",*PATHS}
            for a in d["pred"].values():
                check(a)
    equals(summarize(actual),load(R/"PATH_SUMMARY.json"))
    v.update(raw_sha256=sha(R/"PATH_RAW.json"),cohort_sha256=output["cohort_sha256"],
             runtime_s=output["elapsed"],peak_allocated_mib=output["peak_mib"])
    return v


def pair(z):
    lo,hi=z["interval"]
    return f"{z['total']:+d} 道，{100*z['rate']:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] 个百分点" if lo is not None else "无样本"


def plot(s):
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.55))
    colors=("#992F31","#416A88","#939EA8")
    names=("Full operand cut","Equal-mass operand cut","Equal-mass unrelated cut")
    v=s["repair"]["counts"]
    for j,code in enumerate(("","_capped","_matched")):
        b=axs[0].bar([i+(j-1)*.245 for i in range(3)],[v[i+code]["foil"] for i in PATHS],
                     width=.235,label=names[j],color=colors[j])
        axs[0].bar_label(b,padding=2,fontsize=8)
    axs[0].set_xticks(range(3),["Context","Readout","Joint"])
    axs[0].set_ylabel("Exact +100 reversions")
    axs[0].set_title("K2 after K1 attention edits",loc="left")
    axs[0].legend(frameon=False,fontsize=8)
    med=s["components"]["medians"]
    if med:
        for j,k in enumerate(("joint","joint_capped","joint_matched")):
            q=med[k]
            axs[1].bar([i+(j-1)*.245 for i in range(3)],
                       [q[k+"_shift"] for k in ("propagated","generated_attention","generated_mlp")],
                       width=.235,color=colors[j],label=names[j])
    axs[1].set_xticks(range(3),["Propagated","Attention-generated","MLP-generated"],rotation=11)
    axs[1].set_ylabel("Median change norm")
    axs[1].set_title("K2 physical layer 15",loc="left")
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
        ax.set_ylim(0,max(.05,ax.get_ylim()[1]*1.13))
    fig.tight_layout()
    img=R/"K1_PATHS_K2.png"
    fig.savefig(img,dpi=170,bbox_inches="tight")
    plt.close(fig)
    with Image.open(img) as im:
        assert im.width>1000 and im.height>420
        im.verify()
    return sha(img)


def write(v):
    base=load(R/"NATIVE_SUMMARY.json")
    c=base["confirmation"]
    d=["# 跨全部层和头检验 K1 上下文与读出路径","",
       "在固定冻结的 Ouro-1.4B Base 上，分别切断第一操作数低两位通往上下文和末位读出的注意力。在 K1 使用 24 层的全部查询头。保留原生 V、投影、归一化与后续运算。第二轮以后运行原生算子。","",
       f"300 个确认题中，借位题精确从 K1 +100 错答转为 K2 正答 {c['total']['joint_repairs']}/300，覆盖 {c['distinct_repairs']} 个答案簇。原生同答案无借位对照为 {c['total']['matched_safe_repairs']}/300。"]
    if not base["gate"]:
        d.extend(["","原生门槛未通过。"])
    else:
        s=load(R/"PATH_SUMMARY.json")
        d.extend(["",f"主分析有 {s['repair']['n']} 对自然修复和同答案无借位受体，对齐 {s['repair']['aligned']} 对。两轮读出分差配在预设窗口内的敏感性人群为 {s['margin_matched']['repair']['n']} 对。","",
                  "| 干预 | 借位正答 | 借位精确 +100 | 其他新增错误 | 无借位新增错误 | 同题复位成功 |",
                  "|---|---:|---:|---:|---:|---:|"])
        for name in ARMS:
            b=s["repair"]["counts"][name]
            q=s["safe"]["counts"][name]
            d.append(f"| {name} | {b['correct']} | {b['foil']} | {b['other_new']} | {q['any_new']} | {b['restored']} |")
        d.extend(["","## 对照和区间","",
                  "等量删减在两条前向的每一层、头及查询位置取共同可删去质量，保留各自自然值向量。上下文与读出两臂的查询数及完整切除的总剂量不同。"])
        for p in PATHS:
            z=s["paired"][p]
            dose=s["repair"]["mass"][p]
            d.extend(["",f"- {p} 的查询数中位数 {dose['query_count']:.0f}，等量臂保留完整切除质量的 {100*dose['dose_fraction']:.1f}%。相对于等量无关边，精确回退差为 {pair(z['mass_equalized'])}。借位与无借位的新增错误差为 {pair(z['repair_minus_safe'])}。"])
        for p in ("factorial_reversion","pivot_minus_joint","pivot_minus_context","restore_context_minus_pivot"):
            d.extend(["",f"- {p} 为 {pair(s['paired'][p])}。"])
        d.extend(["","以上为按答案簇配对的 97.5% 区间。二元答案的析因差受输出阈值和饱和影响。"])
        u=s["upper"]
        d.extend(["",f"联合完整切断在 {u['events']}/{u['targets']} 个答案簇产生至少一个精确回退，单侧 97.5% 上界 {100*u['upper_97_5']:.2f}%。",
                  "",f"路径支持门槛：上下文 {'通过' if s['gates']['context'] else '未通过'}，读出 {'通过' if s['gates']['pivot'] else '未通过'}，联合 {'通过' if s['gates']['joint'] else '未通过'}。",
                  "",f"读出占优门槛{'通过' if s['gates']['pivot_dominant'] else '未通过'}。联合精确回退率小于 10% 的排除门槛{'通过' if s['gates']['exclude_joint_10pct'] else '未通过'}。"])
        sub=s["margin_matched"]
        d.extend(["",f"分差敏感性队列的联合完整切除为 {sub['repair']['counts']['joint']['foil']} 个精确回退；等量联合对照的配对差为 {pair(sub['paired']['joint']['mass_equalized'])}。","",
                  "## 原生第 15 层与更深循环","",
                  f"在固定的 {s['components']['n']} 道题上，从实际四角状态重算 K2 第 15 层。图中各项描述物理层响应，与自由续写结果分开。","",
                  "![整答回退和第二轮分量](results/K1_PATHS_K2.png)","",
                  "| Loop | 对齐 / 固定子集 | 原生正答 | 上下文切断正答 | 读出切断正答 | 联合切断正答 |",
                  "|---|---:|---:|---:|---:|---:|"])
        for k,z in s["deeper"].items():
            d.append(f"| K{k} | {z['aligned']}/{z['n']} | "+" | ".join(str(z["arms"][name]["correct"]) for name in ("native",*PATHS))+" |")
        d.extend(["","更深结果只对固定抽取的 K1→K2 自然修复题有效。借位表型、原生状态复位与其他任务的外推应分开。"])
        v["figure_sha256"]=plot(s)
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__=="__main__":
    v=verify()
    write(v)
    save(R/"VERIFICATION.json",v)
    print(v,flush=True)
