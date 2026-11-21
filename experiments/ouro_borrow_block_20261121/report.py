"""Verify the complete native block evidence and render the report and figure."""
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import transformers

import block
import branch_order
import deck
import native_deck
from bidi import unordered
from borrow import needs
from native import MODEL_HASH, REV, load, save, sha
from rule import PREFIX, parse

R=Path("results")


def check(z):
    assert z["tokens"] and parse(z["text"])==z["number"]


def equals(a,b):
    assert json.loads(json.dumps(a))==b


def validate():
    a=load(R/"DECK.json")
    b=load(R/"NATIVE.json")
    assert a["prompt"]==PREFIX and a["previous_sha256"]==sha(deck.PREVIOUS)
    assert b["deck_sha256"]==sha(R/"DECK.json") and len(a["rows"])==len(b["rows"])==260
    seen=deck.history()
    starting=len(seen)
    assert starting==a["earlier_unordered"]
    index={}
    for x,y in zip(a["rows"],b["rows"],strict=True):
        assert x=={k:v for k,v in y.items() if k!="pred"}
        assert x["id"] not in index
        index[x["id"]]=y
        assert x["target"]==x["foil"]-100 and x["split"]==a["groups"][str(x["target"])]
        for key in ("borrow","safe"):
            q=x[key]
            a0,b0=q[0]
            a1,b1=q[-1]
            assert q[1]==[a1,b0] and q[2]==[a0,b1]
            assert a1//100==a0//100 and b1//100==b0//100
            assert [a1-a0,b0-b1]==x[key+"_steps"]
            assert [needs(*z) for z in q]==[False,False,False,key=="borrow"]
            assert b1-a1==x["target"]
            if key=="borrow":
                assert b0-a0==x["foil"] and a1-a0+b0-b1==100
            for z in q:
                w=unordered(z)
                assert w not in seen
                seen.add(w)
        assert len(y["pred"])==2 and all(len(z)==5 for z in y["pred"])
        for seq in y["pred"]:
            for z in seq:
                check(z)
    assert len(seen)-starting==2080
    prior=load(R/"NATIVE_SUMMARY.json")
    equals(native_deck.report(b["rows"],a["groups"]),prior)
    state=dict(status="PASS",model_revision=REV,model_weights_sha256=MODEL_HASH,torch=torch.__version__,
               transformers=transformers.__version__,cuda=torch.version.cuda,python=platform.python_version(),
               device="NVIDIA GeForce RTX 4090",math="pinned BF16 converted to FP32, native eager no-cache, no autocast or TF32",
               arithmetic_prompts=1300,native_sha256=sha(R/"NATIVE.json"),deck_sha256=sha(R/"DECK.json"))
    if not prior["gate"]:
        return state
    nomination=load(R/"LOCK.json")
    assert nomination["natives_sha256"]==state["native_sha256"]
    assert nomination["layer"]==max(block.LAYERS,key=lambda k:nomination["means"][str(k)][1])
    for i,values in nomination["raw"].items():
        assert all(index[q["id"]]["split"]=="discovery" and native_deck.is_repair(index[q["id"]]) for q in values)
        direct=[sum(p["margin_vs_geometry"][j] for p in values)/len(values) for j in (0,1)]
        assert direct==nomination["means"][i]
    records=load(R/"BLOCK_RAW.json")
    plan=load(R/"MATCHING.json")
    assert plan["native_sha256"]==state["native_sha256"]==records["native_sha256"]
    assert records["selection_sha256"]==sha(R/"LOCK.json")
    expected=sorted(x["id"] for x in b["rows"] if x["split"]=="confirmation" and native_deck.is_repair(x))
    assert [r["id"] for r in plan["pairs"]]==expected
    assert [(r["id"],r["kind"]) for r in records["rows"]]==[(x,k) for x in expected for k in ("repair","safe")]
    p={r["id"]:r for r in plan["pairs"]}
    for r in records["rows"]:
        control=r["kind"]=="safe"
        origin=index[r["id"]]
        repl=index[r["source_id"]]
        assert r["source_id"]==(p[r["id"]]["safe"] if control else r["id"])
        assert r["target"]==origin["target"]==repl["target"] and r["foil"]==origin["foil"]
        assert r["difference"]==p[r["id"]]["difference"]
        assert r["margin_matched"]==all(t is not None and t<=2 for t in r["difference"])
        assert r["outcomes"]["native"]==[q[4 if control else 3] for q in repl["pred"]]
        assert set(r["outcomes"])==set(block.MODES)
        for arm in r["outcomes"].values():
            for z in arm:
                check(z)
        assert r["layers"][str(nomination["layer"])]["generated"]==r["outcomes"]["generated"][1]
        if r["aligned"]:
            assert len(r["diagnostic"])==4
            for x in r["diagnostic"].values():
                assert all(0<=q["outside"]<=1 for q in x)
    q=load(R/"BLOCK_SUMMARY.json")
    equals(block.fit(records["rows"]),q)
    state.update(block_sha256=sha(R/"BLOCK_RAW.json"),lock_sha256=sha(R/"LOCK.json"),matching_sha256=sha(R/"MATCHING.json"))
    optional=R/"BRANCH_RAW.json"
    if optional.exists():
        w=load(optional)
        assert w["block_sha256"]==state["block_sha256"]
        for old,new in zip(records["rows"],w["rows"],strict=True):
            assert all(old[k]==new[k] for k in ("id","kind","target","foil"))
            assert new["pred"]["natural"]==old["outcomes"]["native"]
            assert new["pred"]["late"]==new["pred"]["early_fixed_mlp"]==old["outcomes"]["attention"]
            for arm in new["pred"].values():
                for z in arm:
                    check(z)
        equals(branch_order.summarize(w["rows"]),load(R/"BRANCH_SUMMARY.json"))
        state["branch_sha256"]=sha(optional)
    return state


def paired(x):
    lo,hi=x["interval"]
    return f"{x['total']:+d} 例，每目标 {x['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def plot(raw,stats,site):
    rows=[r for r in raw if r["kind"]=="repair" and r["aligned"]]
    xs=np.asarray(block.LAYERS)+1
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,
                         "axes.titlesize":11,"axes.labelsize":10})
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.65),gridspec_kw={"width_ratios":[1.15,1]})
    colors=("#426A8C","#9B372B")
    for k in (0,1):
        for name,ls in (("generated","-"),("propagated","--")):
            val=[np.median([r["diagnostic"][str(l)][k][name] for r in rows]) for l in block.LAYERS]
            axes[0].plot(xs,val,color=colors[k],ls=ls,marker="o",label=f"K{k+1} {name}")
    axes[0].set_title("Readout interaction by physical layer",loc="left",fontweight="semibold")
    axes[0].set_ylabel("Median component norm")
    axes[0].legend(frameon=False,fontsize=8)
    grouped=np.array([stats["per_layer"][str(l)]["K2_foils"] for l in block.LAYERS])
    for j,(name,color) in enumerate((("Remove generated term",colors[1]),("Geometry control","#909BA7"))):
        bars=axes[1].bar(xs+(j-.5)*.32,grouped[:,j],width=.30,label=name,color=color)
        axes[1].bar_label(bars,fontsize=9,padding=2)
    axes[1].set_title("K2 reproduces the original +100",loc="left",fontweight="semibold")
    axes[1].set_ylabel("Exact original wrong integers")
    axes[1].set_ylim(0,max(3,int(grouped.max()*(1.18))+1))
    axes[1].legend(frameon=False,fontsize=8)
    for a in axes:
        a.set_xticks(xs)
        a.set_xlabel("Layer (1-based)")
        a.grid(axis="y",alpha=.17)
        a.set_axisbelow(True)
    for tick in axes[1].get_xticklabels():
        if tick.get_text()==str(site+1):
            tick.set_fontweight("bold")
    fig.tight_layout(pad=1.4)
    path=R/"BLOCK_PROFILE.png"
    fig.savefig(path,dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    with Image.open(path) as im:
        assert im.format=="PNG" and im.width>1000 and im.height>400
        im.verify()


def render():
    b=load(R/"NATIVE_SUMMARY.json")
    s=b["confirmation"]
    d=["# Ouro-1.4B 原生块中新生成的低位交互","",
       "冻结的同一模型在 100 个发现题干和 160 个确认题干上复跑。每个联合借位题都有完整正确答案相同的独立无需借位题。两类四角的数值编辑总量不同，因此另做同向量同剂量对照。","",
       "| 原生条件 | K1 完整正解 | K2 完整正解 |","|---|---:|---:|"]
    for k,v in s["per"].items():
        d.append(f"| {k} | {v['K1_correct']}/{s['n']} | {v['K2_correct']}/{s['n']} |")
    c=s["total"]
    d.extend(["",f"原生自然修复共有 {c['joint_repairs']}/{s['n']}，无需借位组为 {c['matched_safe_repairs']}/{s['n']}。其中 {c['other_three_K1_correct_among_joint_repairs']} 道的其他三个角点在 K1 已完全正确。配对 {paired(s['paired'])}（95%区间）。"])
    if not b["gate"]:
        Path("REPORT_ZH.md").write_text("\n".join(d+["","原生行为门槛未通过。"])+"\n",encoding="utf-8")
        return None
    opt=load(R/"LOCK.json")
    q=load(R/"BLOCK_SUMMARY.json")
    rows=load(R/"BLOCK_RAW.json")["rows"]
    n=q["repair"]["n"]
    site=opt["layer"]
    d.extend(["","## 生成项和传入交互的因果比较","",
              f"发现数据按 K2 诊断分差锁定物理第 {site+1} 层。确认的 {n} 道原生修复均在分岔位置干预。两类均对齐的题有 {q['repair']['aligned']} 组，两个原生分差在固定阈值内的同答案安全匹配有 {q['safe']['matched_margin']} 组。所有原生块和残差写入做了逐位复放。完整整数自由续写。","",
              "| 条件 | K1 完整正解 | K2 精确重现旧错 | 同题双向 | 安全新增 K2 错误 |",
              "|---|---:|---:|---:|---:|"])
    for key,x in q["repair"]["outputs"].items():
        d.append(f"| {key} | {x['K1']}/{n} | {x['K2']}/{n} | {x['joint']}/{n} | {q['safe']['outputs'][key]['K2_harm']}/{q['safe']['n']} |")
    d.extend(["",f"预设双向门槛{'通过' if q['dual_gate'] else '未通过'}。K2 单向特异性门槛{'通过' if q['K2_gate'] else '未通过'}。"])
    if q["dual_gate"] and not q["K2_gate"]:
        d.extend(["","当场生成项对早晚两轮的整答均有因果作用。同剂量几何对照解释不了这些效果。移除上游传入项也能使 K2 回到旧错，当前证据无法把 K2 修复唯一归因于该层当场生成的交互。"])
    for k,v in q["paired"].items():
        d.extend(["",f"{k} 的 K1 / K2 / 双向配对为 {paired(v['K1'])} / {paired(v['K2'])} / {paired(v['joint'])}（97.5%区间）。"])
    a,b=q["repair"]["outputs"]["paired_generated"],q["safe"]["outputs"]["paired_generated"]
    d.extend(["",f"完全相同的借位生成向量在借位受体导致 {a['K2']}/{n} 个精确旧错，在无需借位受体导致 {b['K2']}/{q['safe']['n']} 个；带两类各自匹配几何的双重配对差为 {paired(q['same_dose_context'])}（97.5%区间）。"])
    d.extend(["","## 第 14–17 层的预定次级逐层比较","",
              "在每个物理层单独改动该层输出，由原生后续块及自由整数续写决定完整答案。次级逐层配对区间为 98.75%。","",
              "| 层 | 去除生成项重现旧答 | 匹配几何重现旧答 | 配对差 |","|---|---:|---:|---|"])
    for l in block.LAYERS:
        p=q["per_layer"][str(l)]
        d.append(f"| {l+1} | {p['K2_foils'][0]}/{n} | {p['K2_foils'][1]}/{n} | {paired(p['clustered'])} |")
    d.extend(["","![读出组成和精确回退](results/BLOCK_PROFILE.png)",""])
    valid=[r["diagnostic"][str(site)][1] for r in rows if r["kind"]=="repair" and r["aligned"]]
    d.extend([f"选定层的 K2 加性参考超出四角全前缀范数包络的 token 比例，中位数为 {np.median([p['outside'] for p in valid])*100:.2f}%；读出 token 单独超出包络 {sum(r['readout_outside'] for r in valid)}/{len(valid)}。裁剪变体保留独立结果。",""])
    if (R/"BRANCH_SUMMARY.json").exists():
        post=load(R/"BRANCH_SUMMARY.json")
        cov=post["repair"]
        d.extend(["## 按原生运算顺序干预注意力写入","",
                  "早期改动直接发生在注意力归一化写入处，MLP 可以重算。固定 MLP 对照恢复该位置原生 MLP 写入。该分支的完整生成与晚期仅改动注意力生成分量逐题一致。","",
                  "| 干预 | K1 完整正解 | K2 精确旧错 | 同题双向 | 无借位新增 K2 错误 |",
                  "|---|---:|---:|---:|---:|"])
        for way,r in cov["outcomes"].items():
            d.append(f"| {way} | {r['K1_exact']}/{n} | {r['K2_foil']}/{n} | {r['joint']}/{n} | {post['safe']['outcomes'][way]['K2_harm']}/{post['safe']['n']} |")
        for name,vals in post["comparisons"].items():
            details=" / ".join(f"{val['mean']:+.3f} [{val['interval'][0]:+.3f}, {val['interval'][1]:+.3f}]" for val in vals)
            d.extend(["",f"{name} 的 K1 / K2 配对差（每目标，97.5%区间）为 {details}。"])
    else:
        d.append("按预设触发标准，未启动运算顺序次级检验。")
    d.extend(["","见 METHOD.md 的精确代数、出域诊断和因果对照解释。图表源于逐题输出，完整原始序列、置信度匹配、重算和哈希均随档案。",""])
    Path("REPORT_ZH.md").write_text("\n".join(d),encoding="utf-8")
    plot(rows,q,site)
    return R/"BLOCK_PROFILE.png"


if __name__=="__main__":
    v=validate()
    where=render()
    if where:
        v["block_profile_sha256"]=sha(where)
    save(R/"VERIFICATION.json",v)
    print(v,flush=True)

