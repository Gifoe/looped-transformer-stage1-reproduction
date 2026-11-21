"""Validate the causal experiment and render separate integer and state summaries."""
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
from boundary_deck import PARENT, history
from boundary_stats import ARMS, MODES, summarize
from borrow import needs
from edge import short_hash
from native import MODEL_HASH, REV, load, save, sha
from native_deck import is_repair, report as native_report
from rule import MAX_NEW, PREFIX, parse

R=Path("results")


def check(v):
    assert v["tokens"] and len(v["tokens"])<=MAX_NEW and v["number"]==parse(v["text"])


def equals(a,b):
    assert json.loads(json.dumps(a))==b


def verify():
    d=load(R/"DECK.json")
    n=load(R/"NATIVE.json")
    seen=history()
    assert sha(PARENT)==d["parent_sha256"] and d["earlier_unordered"]==len(seen)
    assert n["deck_sha256"]==sha(R/"DECK.json") and d["prompt"]==PREFIX
    by={}
    for r,t in zip(d["rows"],n["rows"],strict=True):
        assert r=={k:v for k,v in t.items() if k!="pred"} and r["id"] not in by
        by[r["id"]]=t
        y=r["target"]
        assert r["foil"]==y+100 and d["groups"][str(y)]==r["split"]
        for tag in ("borrow","safe"):
            corners=r[tag]
            a0,b0=corners[0]
            a1,b1=corners[3]
            assert corners[1:3]==[[a1,b0],[a0,b1]]
            assert [a1-a0,b0-b1]==r[tag+"_steps"] and b1-a1==y
            assert [needs(*p) for p in corners]==[False,False,False,tag=="borrow"]
            if tag=="borrow":
                assert b0-a0==y+100
        for tag,p in r["donors"].items():
            root=r["safe" if tag=="safe" else "borrow"][3]
            assert all(u//100==v//100 for u,v in zip(p,root,strict=True))
            assert abs(p[r["role"]]-root[r["role"]])==r["delta"] and p[1-r["role"]]==root[1-r["role"]]
            assert needs(*p)==(tag=="same") and p[1]-p[0]==y+(r["delta"] if tag=="opposite" else -r["delta"])
        for p in [*r["borrow"],*r["safe"],*r["donors"].values()]:
            assert all(100<=x<=999 for x in p)
            u=unordered(p)
            assert u not in seen
            seen.add(u)
        assert [len(x) for x in t["pred"]]==[5,5]
        for seq in t["pred"]:
            for a in seq:
                check(a)
    assert len(by)==340 and len(seen)-d["earlier_unordered"]==340*11
    s=load(R/"NATIVE_SUMMARY.json")
    equals(native_report(n["rows"],d["groups"]),s)
    v=dict(status="PASS",model_revision=REV,expected_weights_sha256=MODEL_HASH,
           weights_sha256=os.environ["OURO_ACTUAL_SHA256"],gpu=os.environ["OURO_GPU"],torch=torch.__version__,
           transformers=transformers.__version__,cuda=torch.version.cuda,scored_base_prompts=1700,
           native_sha256=sha(R/"NATIVE.json"),deck_sha256=sha(R/"DECK.json"))
    assert v["weights_sha256"]==MODEL_HASH
    if not s["gate"]:
        return v
    lock=load(R/"COHORT.json")
    z=load(R/"BOUNDARY_RAW.json")
    assert lock["native_sha256"]==z["native_sha256"]==v["native_sha256"] and z["cohort_sha256"]==sha(R/"COHORT.json")
    ids=sorted(r["id"] for r in n["rows"] if r["split"]=="confirmation" and is_repair(r))
    assert [p["id"] for p in lock["pairs"]]==ids
    h=sorted(ids,key=lambda i:short_hash("k2 boundary",i))
    assert lock["diagnostic_ids"]==sorted(h[:24])
    plan={r["id"]:r for r in lock["pairs"]}
    rows=z["rows"]
    assert [(r["id"],r["kind"]) for r in rows]==[(i,k) for i in ids for k in ("repair","safe")]
    for r in rows:
        p=plan[r["id"]]
        repair=r["kind"]=="repair"
        assert r["source_id"]==(r["id"] if repair else p["safe"]) and r["distance"]==p["difference"]
        source=by[r["source_id"]]
        assert all(source[k]==r[k] for k in ("target","foil","delta","role"))
        assert r["native"]==[a[3 if repair else 4] for a in source["pred"]]
        assert len(r["operand"])==6 and r["operand"]==sorted(set(r["operand"]))
        assert set(r["pred"])==set(ARMS) and r["pred"]["native"]==r["native"][1]
        for k in ("opposite","same"):
            a,b=source["donors"][k if repair else "safe"]
            assert r["donor_values"][k]==b-a
            for p in r["donor_native"][k]:
                check(p)
        assert r["pred"]["donor_replay"]==r["donor_native"]["opposite"][1]
        for p in r["pred"].values():
            check(p)
        assert set(r["mass"])=={a for a in ARMS if MODES[a][1]!="live"}
        for a,b in r["mass"].values():
            assert 0<=a<=b+1e-4 and b>0
        if r["diagnostic"]:
            assert repair and r["id"] in lock["diagnostic_ids"]
        assert ("component_diagnostics" in r)==r["diagnostic"]
    equals(summarize(rows),load(R/"BOUNDARY_SUMMARY.json"))
    v.update(raw_sha256=sha(R/"BOUNDARY_RAW.json"),cohort_sha256=z["cohort_sha256"],
             runtime_s=z["elapsed"],peak_allocated_mib=z["peak_allocated_mib"])
    return v


def describe(p):
    lo,hi=p["interval"]
    if lo is None:
        return "无样本"
    return f"{p['total']:+d} 道，{100*p['rate']:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] 个百分点"


def place_values(rows,arm):
    active=[(r,r["pred"][arm]["number"]) for r in rows if r["kind"]=="repair"]
    active=[(r,v) for r,v in active if v is not None and 100<=v<=999]
    ret=[0,0,0,0,0]
    for r,v in active:
        y,d=r["target"],r["donor_values"]["opposite"]
        assert (y+100)//100==d//100 and d%100!=y%100
        facts=(v//100==d//100,v%100==y%100,v%100==d%100,v==100*(y//100)+d%100)
        ret[0]+=1
        for i,x in enumerate(facts,1):
            ret[i]+=x
    return ret


def figure(s):
    r=s["repair"]["arms"]
    a=("native","original_full","opposite_intact","opposite_full","opposite_capped","opposite_matched")
    x=["Native","Cut only","Swap only","Swap + cut","Swap + equal cut","Swap + control"]
    fig,axs=plt.subplots(1,2,figsize=(10.9,3.85),gridspec_kw={"width_ratios":[1.27,1]})
    shades=("#416586","#983C43","#997751","#A8AEB4","#747C84")
    names=("Receiver","Old +100","Opposed donor","Same-borrow donor","Other / unparsed")
    keys=("receiver","old","opposite","same","other")
    bottoms=[0]*len(a)
    for label,k,col in zip(names,keys,shades,strict=True):
        v=[r[z][k]+(r[z]["unparsed"] if k=="other" else 0) for z in a]
        axs[0].bar(range(len(a)),v,bottom=bottoms,color=col,label=label,width=.67)
        bottoms=[u+w for u,w in zip(bottoms,v,strict=True)]
    axs[0].set_xticks(range(len(a)),x,rotation=23,ha="right")
    axs[0].set_ylabel("Complete generated integers")
    axs[0].set_title("K2 answers",loc="left")
    axs[0].legend(frameon=False,fontsize=7,ncol=2)
    c=s["components"]["median"]
    if c:
        parts=("a_write","m_write","output")
        for j,(name,k) in enumerate(zip(("Swap + cut","Swap + equal cut","Swap + control"),
                                        ("opposite_full","opposite_capped","opposite_matched"),strict=True)):
            axs[1].bar([i+(j-1)*.245 for i in range(3)],[c[k][p] for p in parts],
                       width=.235,label=name,color=shades[j])
        axs[1].set_xticks(range(3),["Attention write","MLP write","Block output"])
        axs[1].legend(frameon=False,fontsize=8)
    axs[1].set_ylabel("Median change norm")
    axs[1].set_title("K2 physical layer 15",loc="left")
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y",alpha=.16)
        ax.set_ylim(0,max(.1,ax.get_ylim()[1]*1.18))
    fig.tight_layout()
    dst=R/"BOUNDARY_FACTOR.png"
    fig.savefig(dst,dpi=160,bbox_inches="tight")
    plt.close(fig)
    with Image.open(dst) as img:
        assert img.width>1000 and img.height>430
        img.verify()
    return sha(dst)


def render(v):
    s=load(R/"NATIVE_SUMMARY.json")
    n=s["confirmation"]
    d=["# K1→K2 边界上的上下文状态和操作数读边","",
       "全程用冻结的 Ouro-1.4B Base。对同一个受体，在 K2 入轮保留六个操作数位置的真实 K1 状态，替换或保留其余上下文，同时切断或保留 K2 直接读取操作数位置的全部读边。对随后生成的每个 token 重算同一干预。用完整整数区分受体真值、原 +100 错答和供体答案。","",
       f"300 道确认题中，原生精确 +100→正答为 {n['total']['joint_repairs']}/300，涉及 {n['distinct_repairs']} 个答案簇；同答案的无借位对照为 {n['total']['matched_safe_repairs']}/300。"]
    if s["gate"]:
        z=load(R/"BOUNDARY_SUMMARY.json")
        b=z["repair"]
        a=z["safe"]
        d.extend(["",f"完整生成分析为 {b['n']} 道原生修复与 {a['n']} 道同答案无借位对照。分差均进入匹配阈值的敏感性队列为 {z['margin_matched']['repair']['n']} 对。翻转借位的供体本身正答数为 K1 {b['donor_native']['opposite'][0]}、K2 {b['donor_native']['opposite'][1]}。","",
                  "| 条件 | 受体真值 | 原始 +100 | 翻转借位供体真值 | 同借位供体真值 | 其他整数 | 未解析 | 无借位新增错误 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for arm,u in b["arms"].items():
            ns=[u[k] for k in ("receiver","old","opposite","same","other","unparsed")]+[a["arms"][arm]["new_harm"]]
            d.append(f"| {arm} | "+" | ".join(map(str,ns))+" |")
        raw=load(R/"BOUNDARY_RAW.json")["rows"]
        d.extend(["","将三位输出按百位与低两位拆开作为诊断。供体百位也是原错百位；低两位区分两者。混合答案指受体正确百位与翻转借位供体低两位的拼接。","",
                  "| 条件 | 三位输出 | 供体或原错百位 | 受体低两位 | 供体低两位 | 混合答案 |",
                  "|---|---:|---:|---:|---:|---:|"])
        for arm in ("native","original_full","opposite_intact","opposite_full","opposite_capped","opposite_matched"):
            d.append(f"| {arm} | "+" | ".join(map(str,place_values(raw,arm)))+" |")
        d.extend(["","## 匹配对照和门槛","",
                  "不同上下文中的原始注意力质量可以变化。等质量读边对照在每头、层及查询位置从两条真实前向取共同可删去质量。完全切除和等质量删减对应不同剂量。对称供体配平数值编辑量，改变方向和借位状态。"])
        for arm in ("original_capped","opposite_capped","same_capped"):
            u=b["arms"][arm]
            d.extend(["",f"{arm} 各自输入中可删去的操作数注意力质量，实际删减比例中位数为 {100*u['median_available_fraction']:.1f}%。"])
        for k,y in z["paired"].items():
            d.extend(["",f"- {k} 为 {describe(y)}。"])
        d.extend(["","上述比较为按受体答案簇配对的 97.5% 区间。"])
        for k,w in z["bounds"].items():
            d.extend(["",f"{k} 的事件答案簇为 {w['events']}/{w['n']}，单侧 97.5% 上界 {100*w['upper_97_5']:.2f}%。"])
        for k,x in z["gates"].items():
            d.extend(["",f"{k} 的预设门槛{'通过' if x else '未通过'}。"])
        d.extend(["",f"在固定的 {z['components']['n']} 道题上，另记录原生数字前位置的第 15 层注意力写入、MLP 写入和块输出。范数诊断与整答效果分别判定。","",
                  "![完整答案和物理写入变化](results/BOUNDARY_FACTOR.png)","",
                  "这套干预屏蔽的是 K2 对六个操作数位置的直接读取。替代信息仍可能存在于非操作数位置、残差及未处理的路径。反事实上下文来自自然供体，混合张量本身为干预。"])
        v["figure_sha256"]=figure(z)
    else:
        d.extend(["","原生样本门槛未通过。"])
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__=="__main__":
    v=verify()
    render(v)
    save(R/"VERIFICATION.json",v)
    print(v,flush=True)
