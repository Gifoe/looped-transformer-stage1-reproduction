"""Verify the exact data, interventions, and whole-integer evidence."""
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
from native_deck import is_repair, report as native_report
from rule import MAX_NEW, PREFIX, parse
from value_deck import PARENT, history
from value_stats import ARMS, CONTEXTS, NUMERIC, summarize

R=Path("results")


def equal(x,y):
    assert json.loads(json.dumps(x))==y


def check(x):
    assert x["tokens"] and len(x["tokens"])<=MAX_NEW and parse(x["text"])==x["number"]


def verify():
    d=load(R/"DECK.json")
    n=load(R/"NATIVE.json")
    prior=history()
    assert d["parent_sha256"]==sha(PARENT) and d["earlier_unordered"]==len(prior)
    assert n["deck_sha256"]==sha(R/"DECK.json") and d["prompt"]==PREFIX
    by={}
    for r,t in zip(d["rows"],n["rows"],strict=True):
        assert r=={k:v for k,v in t.items() if k!="pred"} and r["id"] not in by
        by[r["id"]]=t
        y=r["target"]
        assert r["foil"]==y+100 and d["groups"][str(y)]==r["split"]
        for f in ("borrow","safe"):
            qb=r[f]
            ix=r[f+"_donors"]
            (a0,b0),(a,b)=qb[0],qb[3]
            assert qb[1:3]==[[a,b0],[a0,b]]
            assert r[f+"_steps"]==[a-a0,b0-b] and b-a==y
            assert [needs(*p) for p in qb]==[False,False,False,f=="borrow"]
            if f=="borrow":
                assert b0-a0==r["foil"]
            fresh=ix["plus"][0]
            diff=r["delta"]
            e=r[f+"_e"]
            assert fresh==a+(diff if f=="safe" else -diff) and a//100==fresh//100
            assert ix["plus"]==[fresh,b+e] and ix["minus"]==[fresh,b-e]
            assert not needs(fresh,b) and not needs(*ix["plus"])
            assert needs(*ix["minus"])==(f=="borrow")
            mix=r["mix" if f=="borrow" else "safe_mix"]
            assert mix==b-fresh and len({y,y+100,mix,mix-e,mix+e})==5
            for p in [*qb,*ix.values()]:
                assert all(100<=v<=999 for v in p)
                u=unordered(p)
                assert u not in prior
                prior.add(u)
        assert [len(x) for x in t["pred"]]==[5,5]
        for g in t["pred"]:
            for p in g:
                check(p)
    assert len(by)==340 and len(prior)-d["earlier_unordered"]==4080
    summary=load(R/"NATIVE_SUMMARY.json")
    equal(native_report(n["rows"],d["groups"]),summary)
    audit=dict(status="PASS",model_revision=REV,expected_weights_sha256=MODEL_HASH,
               weights_sha256=os.environ["OURO_ACTUAL_SHA256"],gpu=os.environ["OURO_GPU"],
               torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
               native_sha256=sha(R/"NATIVE.json"),deck_sha256=sha(R/"DECK.json"),scored_base_prompts=1700)
    assert audit["weights_sha256"]==MODEL_HASH
    if not summary["gate"]:
        return audit
    lock=load(R/"COHORT.json")
    z=load(R/"VALUE_RAW.json")
    assert lock["native_sha256"]==z["native_sha256"]==audit["native_sha256"]
    assert z["cohort_sha256"]==sha(R/"COHORT.json")
    expected=sorted(r["id"] for r in n["rows"] if r["split"]=="confirmation" and is_repair(r))
    assert [r["id"] for r in lock["pairs"]]==expected
    h=sorted(expected,key=lambda i:short_hash("value-read",i))
    assert lock["diagnostic_ids"]==sorted(h[:24])
    plan={r["id"]:r for r in lock["pairs"]}
    rows=z["rows"]
    assert [(r["id"],r["kind"]) for r in rows]==[(i,k) for i in expected for k in ("repair","safe")]
    for r in rows:
        p=plan[r["id"]]
        repair=r["kind"]=="repair"
        assert r["source_id"]==(r["id"] if repair else p["safe"]) and r["distance"]==p["difference"]
        s=by[r["source_id"]]
        f="borrow" if repair else "safe"
        ix=s[f+"_donors"]
        assert r["mixed"]==s[f][3][1]-ix["plus"][0]
        assert r["plus"]==ix["plus"][1]-ix["plus"][0]
        assert r["minus"]==ix["minus"][1]-ix["minus"][0]
        assert r["target"]==s["target"] and r["foil"]==s["foil"]
        assert r["native"]==[a[3 if repair else 4] for a in s["pred"]]
        assert set(r["pred"])==set(ARMS) and r["pred"]["receiver_native"]==r["native"][1]
        assert len(r["operand"])==6 and r["operand"]==sorted(set(r["operand"]))
        for v in r["pred"].values():
            check(v)
        for a in r["donor_native"].values():
            for v in a:
                check(v)
        if r["diagnostic"]:
            assert repair and r["id"] in lock["diagnostic_ids"]
        assert ("diagnostics" in r)==r["diagnostic"]
    equal(summarize(rows),load(R/"VALUE_SUMMARY.json"))
    audit.update(raw_sha256=sha(R/"VALUE_RAW.json"),cohort_sha256=z["cohort_sha256"],
                 runtime_s=z["elapsed"],peak_allocated_mib=z["peak_allocated_mib"])
    return audit


def phrase(d):
    a,b=d["interval"]
    if a is None:
        return "无样本"
    return f"{d['total']:+d} 道，{d['rate']*100:+.1f} [{a*100:+.1f}, {b*100:+.1f}] 个百分点"


def lookup_audit(lines,proof):
    from lookup import ARMS as options, dataset
    from origin_stats import paired
    d,native,summary=[load(R/p) for p in ("LOOKUP_DECK.json","LOOKUP_RAW.json","LOOKUP_SUMMARY.json")]
    assert d==dataset() and summary["deck_sha256"]==sha(R/"LOOKUP_DECK.json")
    assert summary["raw_sha256"]==sha(R/"LOOKUP_RAW.json")
    for a,b in zip(d["rows"],native["native"],strict=True):
        assert a=={k:v for k,v in b.items() if k not in ("r_native","d_native")}
        for arm in ("r_native","d_native"):
            for v in b[arm]:
                check(v)
    base=[sum(r["r_native"][k]["number"]==r["target"] for r in native["native"]) for k in (0,1)]
    assert base==summary["receiver_native"] and summary["gate"]==(base[1]>=64)
    lines.extend(["","## 独立键值查表任务","",
                  f"固定 128 对独立表格。受体原生 K1 / K2 正答 {base[0]}/{base[1]}。混合终点为受体表中与供体查询键配对的数值。"])
    if summary["gate"]:
        rows=native["interventions"]
        cohort=[r["id"] for r in native["native"] if r["r_native"][1]["number"]==r["target"]]
        assert [r["id"] for r in rows]==cohort and len(rows)==summary["n_intervened"]
        source={r["id"]:r for r in native["native"]}
        for r in rows:
            s=source[r["id"]]
            assert all(r[k]==s[k] for k in ("target","mixed","copied"))
            assert set(r["pred"])==set(options) and r["pred"]["native"]==s["r_native"][1]
            for a in r["pred"].values():
                check(a)
        for a in options:
            assert summary["counts"][a]=={k:sum(x["pred"][a]["number"]==x[k] for x in rows)
                                          for k in ("target","mixed","copied")}
        flags=[int(r["pred"]["donor"]["number"]==r["mixed"])-int(r["pred"]["shuffle"]["number"]==r["mixed"]) for r in rows]
        equal(paired(rows,flags),summary["paired"])
        c=summary["counts"]
        lines.extend(["",f"自然 K2 正答队列有 {len(rows)} 道。真 V 与打乱 V 分别得到 {c['donor']['mixed']} 和 {c['shuffle']['mixed']} 道混合正答，配对差 {phrase(summary['paired'])}。真 V 直接复述供体整答 {c['donor']['copied']} 道。预设迁移门槛{'通过' if summary['supported'] else '未通过'}。"])
    else:
        assert not native["interventions"]
        lines.extend(["","查表原生正答门槛未通过。"])
    proof["lookup_raw_sha256"]=summary["raw_sha256"]


def plot(z):
    r=z["repair"]["arms"]
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.45))
    colors=("#607C92","#9A4B34","#A3ADB4")
    for j,(k,name) in enumerate((("native","Native"),("donor","Donor V"),("shuffle","Matched shuffle"))):
        bars=axes[0].bar([i+(j-1)*.245 for i in range(3)],[r[a+"_"+k]["mixed"] for a in CONTEXTS],
                         width=.23,color=colors[j],label=name)
        axes[0].bar_label(bars,padding=1,fontsize=8)
    labels=("Receiver","Opposed","Same borrow")
    axes[0].set_xticks(range(3),labels)
    axes[0].set_ylabel("Exact mixed answers")
    axes[0].set_title("K2 composition by incoming context",loc="left")
    axes[0].legend(frameon=False,fontsize=8)
    c=z["diagnostic"]["median"]
    if c:
        for j,k in enumerate(("a_write","m_write","output")):
            axes[1].bar([i+(j-1)*.245 for i in range(3)],[c[a+"_donor"][k] for a in CONTEXTS],
                        width=.23,color=colors[j],label=("Attention write","MLP write","Block output")[j])
        axes[1].legend(frameon=False,fontsize=8)
    axes[1].set_xticks(range(3),labels)
    axes[1].set_ylabel("Median change norm")
    axes[1].set_title("Layer 15 diagnostic",loc="left")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0,max(.05,ax.get_ylim()[1]*1.18))
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
    fig.tight_layout()
    p=R/"VALUE_COMPOSE.png"
    fig.savefig(p,dpi=170,bbox_inches="tight")
    plt.close(fig)
    with Image.open(p) as img:
        assert img.width>1000 and img.height>420
        img.verify()
    return sha(p)


def report(audit):
    n=load(R/"NATIVE_SUMMARY.json")
    c=n["confirmation"]
    d=["# 在原生注意力权重下交换 K2 操作数 V","",
       "冻结 Ouro-1.4B Base。对每个上下文，先获得未干预 K2 注意力权重，再逐层保持这些权重，只交换第一操作数的 V，固定第二操作数各层为该受体背景下原始 V。每个输出 token 重算同一干预，完整整数区分受体答案、原 +100 错答、混合算式和整道供体答案。","",
       f"300 道确认题中有 {c['total']['joint_repairs']} 道原生精确 +100→正答，分布于 {c['distinct_repairs']} 个答案簇。同答案无借位组为 {c['total']['matched_safe_repairs']}/300。"]
    if not n["gate"]:
        d.extend(["","原生门槛未通过。"])
    else:
        z=load(R/"VALUE_SUMMARY.json")
        r,s=z["repair"],z["safe"]
        d.extend(["",f"完整分析共 {r['n']} 道自然修复与 {s['n']} 道无借位对照。受体分差匹配敏感性队列有 {z['margin_matched']['repair']['n']} 对。翻借位供体自己的原生整答为 K1 {r['donor_native']['opposite'][0]}、K2 {r['donor_native']['opposite'][1]}。","",
                  "| 条件 | 受体正答 | 原始 +100 | 混合正答 | 翻借位供体 | 同借位供体 | 其他 / 未解析 | 无借位混合正答 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for a,b in r["arms"].items():
            v=[b[k] for k in NUMERIC]+[b["other"]+b["unparsed"],s["arms"][a]["mixed"]]
            d.append(f"| {a} | "+" | ".join(map(str,v))+" |")
        d.extend(["","## 预设检验","",
                  "配对区间为按受体答案簇的 97.5% bootstrap。打乱 V 位移在每个 KV 头及操作数 token 保持范数和沿原 V 的投影，接收方的输出路由仍使用同一背景的正常权重。"])
        for k,p in z["paired"].items():
            d.extend(["",f"- {k} 为 {phrase(p)}。"])
        for k,p in z["bounds"].items():
            d.extend(["",f"{k} 中真 V 相比打乱 V 的额外混合命中出现在 {p['events']}/{p['n']} 个答案簇，单侧 97.5% 上界 {100*p['upper_97_5']:.2f}%。"])
        for k,v in z["gates"].items():
            d.extend(["",f"{k} 的预设门槛{'通过' if v else '未通过'}。"])
        d.extend(["",f"固定诊断子集有 {z['diagnostic']['n']} 道。比较了实际供体 A 的 V 是否随其 B 改变，逐层留存原收受体 B 的 V，并校验固定权重的无变更回放。图中范数来自层 15 数字生成前位置。","",
                  "![整答结果与分支诊断](results/VALUE_COMPOSE.png)","",
                  "三种上下文的注意力权重各取自身的未干预 K2 前向。所有算术混合终点均无需借位，因此整答证据约束可组合的数值读取。对新借位运算的形成和 K2 相对 K1 的独有作用仍需独立检验。固定权重的反事实也保留了正常前向的路由，后续查询可随 V 变化的情况另作判断。"])
        audit["figure_sha256"]=plot(z)
        if (R/"LOOKUP_SUMMARY.json").exists():
            lookup_audit(d,audit)
        elif not z["gates"]["composition"]:
            d.extend(["","算术的预设组合门槛未通过，独立查表检验按锁定条件未触发。"])
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__=="__main__":
    v=verify()
    report(v)
    save(R/"VERIFICATION.json",v)
    print(v,flush=True)
