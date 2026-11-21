"""Audit the fresh cohort, all integer outcomes, and causal comparisons."""
import json
import os
from pathlib import Path
import platform
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import transformers

from bidi import unordered
from borrow import needs
from native import MODEL_HASH, REV, load, save, sha
from native_deck import is_repair, report as native_report
from origin_deck import PARENT, history
from origin_stats import ARMS, SOURCES, summarize
from rule import PREFIX, parse

R = Path("results")


def equivalent(a,b):
    assert json.loads(json.dumps(a))==b


def check(answer):
    assert answer["tokens"] and parse(answer["text"])==answer["number"]


def validate():
    deck = load(R/"DECK.json")
    native = load(R/"NATIVE.json")
    seen, prior = history()
    assert len(seen)==deck["earlier_unordered"] and deck["parent_sha256"]==sha(PARENT)
    assert deck["prompt"]==PREFIX and native["deck_sha256"]==sha(R/"DECK.json")
    by = {}
    for a,b in zip(deck["rows"],native["rows"],strict=True):
        assert a=={k:v for k,v in b.items() if k!="pred"}
        assert a["id"] not in by and a["target"] not in prior
        assert deck["groups"][str(a["target"])]==a["split"] and a["foil"]==a["target"]+100
        by[a["id"]] = b
        for key in ("borrow","safe"):
            q = a[key]
            a0,b0 = q[0]
            a1,b1 = q[-1]
            assert [a1-a0,b0-b1]==a[key+"_steps"] and q[1:3]==[[a1,b0],[a0,b1]]
            assert a1//100==a0//100 and b1//100==b0//100
            assert [needs(*s) for s in q]==[False,False,False,key=="borrow"]
            assert b1-a1==a["target"]
            if key=="borrow":
                assert b0-a0==a["foil"]
            for p in q:
                u = unordered(p)
                assert u not in seen
                seen.add(u)
        assert len(b["pred"])==2 and all(len(s)==5 for s in b["pred"])
        for branch in b["pred"]:
            for answer in branch:
                check(answer)
    orig = load(R/"NATIVE_SUMMARY.json")
    equivalent(native_report(native["rows"],deck["groups"]),orig)
    ver = dict(status="PASS",model_revision=REV,model_weights_sha256=MODEL_HASH,
               python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
               driver_runtime=torch.version.cuda,device=os.environ.get("OURO_GPU","unspecified"),
               native_sha256=sha(R/"NATIVE.json"),deck_sha256=sha(R/"DECK.json"),scored=5*len(by))
    if not orig["gate"]:
        return ver
    raw = load(R/"ORIGIN_RAW.json")
    assert raw["native_sha256"]==ver["native_sha256"]
    cases = sorted(x["id"] for x in by.values() if x["split"]=="confirmation" and is_repair(x))
    assert [(r["id"],r["kind"]) for r in raw["rows"]]==[(x,k) for x in cases for k in ("repair","safe")]
    for r in raw["rows"]:
        p = by[r["id"]]
        s = by[r["source_id"]]
        ix = 3 if r["kind"]=="repair" else 4
        assert (r["target"],r["foil"])==(p["target"],p["foil"])==(s["target"],s["foil"])
        assert r["native"]==[branch[ix] for branch in s["pred"]]
        assert r["pred"]["clean"]==r["native"][1] and set(r["pred"])==set(ARMS)
        for a in r["pred"].values():
            check(a)
        if r["aligned"]:
            assert set(r["shifts"])==set(SOURCES)
            assert all(r["shifts"][k][a]==0. for k in ("joint","quarter") for a in ("A_shift","M_shift"))
            assert all(r["first_token"][a] in p["tokens"] for a,p in r["pred"].items())
        else:
            assert all(a==r["pred"]["clean"] for a in r["pred"].values())
    for a,b in zip(raw["rows"][::2],raw["rows"][1::2],strict=True):
        assert a["aligned"]==b["aligned"] and a.get("source_norm")==b.get("source_norm")
    equivalent(summarize(raw["rows"]),load(R/"ORIGIN_SUMMARY.json"))
    ver.update(origin_sha256=sha(R/"ORIGIN_RAW.json"),runtime_s=raw["elapsed"],peak_allocated_mib=raw["peak_allocated_mib"])
    return ver


def pct(p):
    return f"{100*p:+.1f}" if p is not None else "NA"


def contrast(v):
    return f"{v['total']:+d} 道，{pct(v['rate'])} [{pct(v['interval'][0])}, {pct(v['interval'][1])}] 个百分点"


def figure(stats):
    rep = stats["repair"]["counts"]
    fig, axes = plt.subplots(1,2,figsize=(9.5,3.8))
    pairs = (("Symmetric K1 write",("symmetric","source_geometry")),
             ("Joint-only K1 write",("joint","quarter")),
             ("Equal-dose clean K2",("clean_dose_p","clean_dose_m")))
    palette = ("#326485","#AE5134")
    for k in (0,1):
        b = axes[0].bar([i+(k-.5)*.32 for i in range(len(pairs))],
                        [rep[p[1][k]]["foil"] for p in pairs],width=.3,color=palette[k],
                        label=("Actual / full / incoming","Geometry / quarter / MLP")[k])
        axes[0].bar_label(b,padding=2,fontsize=9)
    axes[0].set_xticks(range(len(pairs)),[p[0] for p in pairs],rotation=16,ha="right")
    axes[0].set_ylabel("Exact +100 reversions")
    axes[0].set_title("Source edits and K2 removals",loc="left")
    axes[0].legend(frameon=False,fontsize=8)
    for k,post in enumerate(("","_geometry")):
        labels = ["symmetric_restore_"+x+post for x in ("p","m","both")]
        b = axes[1].bar([i+(k-.5)*.32 for i in range(3)],
                       [rep[a]["source_restored"] for a in labels],
                       width=.3,color=palette[k],label=["Component","Geometry"][k])
        axes[1].bar_label(b,padding=2,fontsize=9)
    axes[1].set_xticks(range(3),["Incoming","Generated MLP","Both"])
    axes[1].set_ylabel("Same-question full-integer rescues")
    axes[1].set_title("Restore after symmetric K1 edit",loc="left")
    axes[1].legend(frameon=False,fontsize=8)
    for a in axes:
        for side in ("right","top"):
            a.spines[side].set_visible(False)
        a.set_ylim(0,max(3,a.get_ylim()[1]*1.16))
        a.grid(axis="y",alpha=.18)
        a.set_axisbelow(True)
    fig.tight_layout()
    path=R/"CROSS_LOOP_ORIGIN.png"
    fig.savefig(path,dpi=160,bbox_inches="tight")
    plt.close(fig)
    with Image.open(path) as im:
        assert im.width>=1000 and im.height>=400
        im.verify()
    return sha(path)


def render(verified):
    all_rows=load(R/"NATIVE.json")["rows"]
    orig=load(R/"NATIVE_SUMMARY.json")
    s=orig["confirmation"]
    c=s["total"]
    d=["# 第 15 层 K1 MLP 写入对 K2 原生借位纠错的作用","",
       "上一轮预先锁定物理第 15 层与 K1 的归一化 MLP 写入。本轮从独立题目取四角，干预一次真实写入，保留后续原生计算。生成结果按完整整数判定。","",
       f"独立确认 {s['n']} 个题干。原生 K1 恰好多 100、K2 答对为 {c['joint_repairs']}/{s['n']}，同答案无借位组为 {c['matched_safe_repairs']}/{s['n']}。完整的两轮行为分层见下表。","",
       "| 确认条件 | 错→对 | 对→错 | 对→对 | 错→错 |","|---|---:|---:|---:|---:|"]
    transitions = ((False,True),(True,False),(True,True),(False,False))
    for i,name in enumerate(("借位四角 00","10","01","11","无借位 11")):
        totals=[0]*4
        for r in all_rows:
            if r["split"]=="confirmation":
                a,b=(r["borrow"]+[r["safe"][3]])[i]
                old,new=[p[i]["number"]==b-a for p in r["pred"]]
                totals[transitions.index((old,new))] += 1
        d.append(f"| {name} | "+" | ".join(map(str,totals))+" |")
    if not orig["gate"]:
        d.extend(["","原生门槛未通过，按锁定规则停止因果干预。"])
    else:
        stats=load(R/"ORIGIN_SUMMARY.json")
        rep,safe=stats["repair"],stats["safe"]
        raw=load(R/"ORIGIN_RAW.json")["rows"]
        by={r["id"]:r for r in all_rows}
        width=sum(r["kind"]=="safe" and by[r["source_id"]]["safe"][3][1]>=1000 for r in raw)
        same_width={b["id"] for b in raw if b["kind"]=="safe" and by[b["source_id"]]["safe"][3][1]<1000}
        width_stats=summarize([r for r in raw if r["id"] in same_width])
        d.extend(["",f"因果队列有 {rep['n']} 对借位与无借位受体，读出处可对齐 {rep['aligned']} 对，两个原生分差均在固定匹配阈值内 {rep['matched']} 对。无借位受体有 {width} 道为四位操作数。","",
                  "| 条件 | 借位整答正确 | 借位精确 +100 | 回退后同题救回 | 无借位精确 +100 | 无借位新增其他错答 |",
                  "|---|---:|---:|---:|---:|---:|"])
        for a in ARMS:
            r=rep["counts"][a]
            t=safe["counts"][a]
            d.append(f"| {a} | {r['correct']} | {r['foil']} | {r['source_restored']}/{r['source_losses']} | {t['foil']} | {t['new_nonfoil']} |")
        d.extend(["","## 配对判别与因果边界","",
                  "所有区间均为按正确答案分组的配对 97.5% bootstrap 区间。指标以题干比例计算。同题救回要求原生正答、改 K1 后精确 +100、复位后再次完整答对。","",
                  f"传入项的 K1 来源门槛{'通过' if stats['gates']['p'] else '未通过'}。当轮 MLP 生成项的 K1 来源门槛{'通过' if stats['gates']['m'] else '未通过'}。"])
        for k,q in stats["source_contrasts"].items():
            if isinstance(q,dict):
                d.extend(["",f"{k} 为 {contrast(q)}。"])
        d.extend(["",f"同三位数宽度受体的同向量对照为 {contrast(width_stats['source_contrasts']['symmetric_same_vector_borrow_minus_safe'])}。"])
        for bg in ("clean","symmetric"):
            z=stats[bg+"_factorial"]
            d.extend(["",f"{bg} 背景的同剂量比较中，传入项占优门槛{'通过' if z['dominance']['p'] else '未通过'}，当轮 MLP 项占优门槛{'通过' if z['dominance']['m'] else '未通过'}。"])
            for k in ("p_vs_m","p_vs_geometry","m_vs_geometry","discrete_interaction"):
                d.append(f"- {k} 为 {contrast(z[k])}。")
        d.extend(["","只改联合角时，另外三角的 K2 输入逐位相同，K2 注意力与 MLP 生成项逐位相同。分摊改动后，多个角进入 K2 的状态变化允许加性参考位移。仅联合角的传播复位在构造上恢复该位置的净差分，不能独立证明唯一中介。四角等范数几何匹配约束联合角的写入轴，其他角的绝对状态仍可以变化。","",
                  "| K1 源写入 | K2 传入项变化范数中位数 | K2 注意力新生成项 | K2 MLP 新生成项 |",
                  "|---|---:|---:|---:|"])
        for a,q in rep["median_response"].items():
            d.append(f"| {a} | {q['P_shift']:.4f} | {q['A_shift']:.4f} | {q['M_shift']:.4f} |")
        aligned=[r for r in raw if r["kind"]=="repair" and r["aligned"]]
        if aligned:
            d.extend(["","正确首 token 减原错首 token 的 logit 分差只作诊断。相对原生 K2 的逐题分差变化中位数如下。","",
                      "| 情况 | 分差变化 |","|---|---:|"])
            for arm in ("joint","symmetric","source_geometry","symmetric_restore_p","symmetric_restore_m",
                        "symmetric_restore_both","clean_dose_p","clean_dose_m"):
                margin=median(r["margin"][arm]-r["margin"]["clean"] for r in aligned)
                d.append(f"| {arm} | {margin:+.3f} |")
        d.extend(["","此干预只评估筛选出的自然 K1→K2 借位修复。真正的跨任务泛化、K3/K4 迁移和全部 K1 物理写入的来源鉴定仍需另证。","",
                  "![K1 来源干预与 K2 复位](results/CROSS_LOOP_ORIGIN.png)"])
        verified["figure_sha256"]=figure(stats)
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__=="__main__":
    verified=validate()
    render(verified)
    save(R/"VERIFICATION.json",verified)
    print(verified,flush=True)
