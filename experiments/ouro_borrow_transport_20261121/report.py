"""Verify complete decoded evidence and reproduce the registered summaries."""
from collections import Counter
import json
from pathlib import Path
import platform

import numpy as np
import torch
import transformers

import baseline
import borrow
import mechanism
import readout_check
from bidi import unordered
from native import MODEL_HASH, REV, load, save, sha
from rule import PREFIX, parse

ROOT=Path("results")


def compare(a,b):
    assert json.loads(json.dumps(a))==b


def integer_record(z):
    assert z["tokens"] and parse(z["text"])==z["number"]


def all_pred(p):
    for x in p.values():
        assert len(x)==2
        for z in x:
            integer_record(z)


def verify():
    design=load(ROOT/"DECK.json")
    actual=load(ROOT/"NATIVE.json")
    assert design["prompt"]==PREFIX and design["prior_sha256"]==[sha(x) for x in (*borrow.INPUTS,baseline.EXTRA)]
    assert actual["deck_sha256"]==sha(ROOT/"DECK.json") and len(design["rows"])==len(actual["rows"])==435
    old=baseline.history()
    assert len(old)==design["historical_unique"]
    idx={}
    count=Counter()
    for a,b in zip(design["rows"],actual["rows"],strict=True):
        assert a=={k:v for k,v in b.items() if k!="pred"}
        assert a["foil"]==a["target"]+100 and a["split"]==design["targets"][str(a["target"])]
        count[a["split"],a["target"]//100]+=1
        for t,(x,y) in enumerate(b["pairs"]):
            assert y-x==a["target"] and borrow.needs(x,y)==bool(t) and unordered((x,y)) not in old
            assert (205<=x<300) if a["split"]=="donors" else (105<=x<205)
            old.add(unordered((x,y)))
        assert len(b["pred"])==2 and all(len(q)==2 for q in b["pred"])
        for q in b["pred"]:
            for z in q:
                integer_record(z)
        assert a["id"] not in idx
        idx[a["id"]]=b
    assert all(count[s,h]==n for (s,n) in (("donors",39),("receivers",48)) for h in range(2,7))
    reference=load(ROOT/"NATIVE_SUMMARY.json")
    compare(baseline.summary(actual["rows"],design["targets"]),reference)
    result=dict(status="PASS",model_revision=REV,model_sha256=MODEL_HASH,python=platform.python_version(),
                torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
                gpu="NVIDIA GeForce RTX 4090",math="pinned BF16 checkpoint promoted to FP32; native eager no-cache; no TF32 or autocast",
                new_unordered_operand_pairs=len(old)-design["historical_unique"],
                deck_sha256=sha(ROOT/"DECK.json"),native_sha256=sha(ROOT/"NATIVE.json"))
    m=ROOT/"MATCHING.json"
    if not m.exists():
        assert not reference["gate"]
        save(ROOT/"VERIFICATION.json",result)
        return result
    plan=load(m)
    tok=transformers.AutoTokenizer.from_pretrained("/root/autodl-tmp/ouro-cross-task-20261118/model",
                                                  trust_remote_code=True,local_files_only=True)
    _,bank_plan,expected=mechanism.build_plan(tok,actual["rows"])
    assert plan["plan"]==[dict(id=q["id"],disjoint=q["disjoint"],aligned=q["aligned"]) for q in expected]
    assert plan["donor_ready"]==len(bank_plan)
    assert plan["native_sha256"]==result["native_sha256"]
    who=sorted(x["id"] for x in actual["rows"] if x["split"]=="receivers" and borrow.native_event(x,1))
    assert [x["id"] for x in plan["plan"]]==who
    for r in plan["plan"]:
        pair=idx[r["id"]]
        targets={pair["target"]//100,pair["foil"]//100}
        for k in ("disjoint","aligned"):
            if r[k] is None:
                continue
            d=idx[r[k]]
            units={d["target"]//100,d["foil"]//100}
            assert d["split"]=="donors" and borrow.native_event(d,1) and baseline.exact_safe(d)
            assert (targets==units) if k=="aligned" else targets.isdisjoint(units)
    if not plan["gate"]:
        save(ROOT/"VERIFICATION.json",result)
        return result
    base=ROOT/"SUBSPACE.json"
    sub=load(base)
    assert sub["rank"]==8 and all(idx[q]["split"]=="donors" for q in sub["donor_ids"])
    mat=np.array(sub["basis_vectors"])
    assert mat.shape==(8,2048)
    np.testing.assert_allclose(mat@mat.T,np.eye(8),rtol=2e-5,atol=2e-5)
    raw=ROOT/"MECHANISM_RAW.json"
    tracked=load(raw)
    assert tracked["native_sha256"]==result["native_sha256"] and tracked["matching_sha256"]==sha(m)
    assert tracked["subspace_sha256"]==sha(base)
    assert [(r["id"],r["kind"]) for r in tracked["rows"]]==[(w,k) for w in who for k in ("repair","safe")]
    self_answers={}
    for row in tracked["rows"]:
        item=idx[row["id"]]
        assert row["target"]==item["target"] and row["foil"]==item["foil"]
        original=[z[row["kind"]=="repair"] for z in item["pred"]]
        assert row["pred"]["native"]==original
        for k in ("self_query_restore","self_A_restore","distant_restore"):
            assert row["pred"][k]==original
        if row["kind"]=="repair":
            self_answers[row["id"]]=row["pred"]["self_query"]
        all_pred(row["pred"])
    compare(mechanism.summarize(tracked["rows"]),load(ROOT/"MECHANISM_SUMMARY.json"))
    result.update(matching_sha256=sha(m),subspace_sha256=sha(base),mechanism_sha256=sha(raw))
    r=ROOT/"READOUT_RAW.json"
    out=load(r)
    bank=load(ROOT/"DISJOINT_BASES.json")
    assert out["matching_sha256"]==sha(m) and out["bases_sha256"]==sha(ROOT/"DISJOINT_BASES.json")
    for h,fit in bank.items():
        support={int(h),int(h)+1}
        for d in fit["ids"]:
            ref=idx[d]
            assert ref["split"]=="donors" and support.isdisjoint({ref["target"]//100,ref["foil"]//100})
        if fit["eligible"]:
            mat=np.array(fit["vectors"])
            assert mat.shape==(8,2048)
            np.testing.assert_allclose(mat@mat.T,np.eye(8),rtol=2e-5,atol=2e-5)
    assert [x["id"] for x in out["rows"]]==who
    for row in out["rows"]:
        a=idx[row["id"]]
        assert row["pred"]["native"]==[x[1] for x in a["pred"]]
        assert row["pred"]["self"]==self_answers[row["id"]]
        assert row["basis_ready"]==bank[str(row["target"]//100)]["eligible"]
        all_pred(row["pred"])
    compare(readout_check.summarize(out["rows"]),load(ROOT/"READOUT_SUMMARY.json"))
    result.update(readout_sha256=sha(r),disjoint_bases_sha256=sha(ROOT/"DISJOINT_BASES.json"))
    save(ROOT/"VERIFICATION.json",result)
    return result


def fmt(e):
    a,b=e["interval"]
    return f"{e['total']:+d} 个配对净变化。每目标均值 {e['mean']:+.3f}，区间 [{a:+.3f}, {b:+.3f}]。"


def render():
    original=load(ROOT/"NATIVE_SUMMARY.json")
    d=["# Ouro-1.4B Base 自然漏借位路径的跨数字迁移检验","",
       "固定检查点和完整词表贪心生成。供受体正确目标整数和首操作数池分离。每对受控题共享完整正确整数和 +100 错值。",
       "", "| 群组 | 条件 | K1 完整正确 | K2 完整正确 | 精确漏借位后修复 |","|---|---|---:|---:|---:|"]
    for split,name in (("donors","供体库"),("receivers","独立受体")):
        for key,condition in (("borrow","需向百位借位"),("safe","无需借位")):
            x=original[split][key]
            d.append(f"| {name} | {condition} | {x['K1_correct']}/{x['n']} | {x['K2_correct']}/{x['n']} | {x['exact_repair']}/{x['n']} |")
    if not original["gate"]:
        d.extend(["","原生行为门槛未通过。"])
        Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")
        return
    p=load(ROOT/"MATCHING.json")
    d.extend(["",f"自然纠错受体有 {p['n']} 道，找到两种百位数字均互斥的供体 {p['n_disjoint']} 道，找到百位标签相同但完整答案不同的供体 {p['n_aligned']} 道。合格的供体题对有 {p['donor_ready']} 个。"])
    if not p["gate"]:
        d.extend(["","匹配门槛未通过。"])
        Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")
        return
    s=load(ROOT/"MECHANISM_SUMMARY.json")
    n=s["repair"]["n"]
    d.extend(["","## 第 14 层查询在受体 K/V 上的首操作数效应","",
               f"固定受体自身的 K 和 V，在输出投影前对子路径的 AV 位移做同剂量移植。表内双向要求同题 K1 完整正解且 K2 重现该受体自己原来的精确错答。另有 {original['receivers']['exact_repair_and_safe_double_correct']} 道匹配对照原生两轮都正确。新增错误以对应轮原本正确的题为条件。",
               "", "| 干预 | K1 答对 | K2 恰好多 100 | 同题双向 | 新增匹配对照错误 K1/K2 |",
               "|---|---:|---:|---:|---:|"])
    for m,x in s["repair"]["outcomes"].items():
        a,b=s["safe"]["outcomes"][m]["new_wrong"]
        d.append(f"| {m} | {x['K1_correct']}/{n} | {x['K2_foil']}/{n} | {x['joint']}/{n} | {a}/{b} |")
    d.extend(["",f"跨数字借位供体门槛{'通过' if s['transport_gate'] else '未通过'}。"])
    for k,v in s["transport_comparisons"].items():
        d.extend(["",f"跨数字借位供体相对 {k} 的双向差为 {fmt(v)}（95%）。"])
    energy=load(ROOT/"SUBSPACE.json")["learn_energy"]
    d.extend(["","## 第 17 层读出复位和子空间检验","",
               "整体复位在实际下一词元和末位置输出状态上恢复原生读出。该操作是结构校验。较窄的因果比较只去除或保留独立供体库学习的八维投影，每个随机控制匹配其向量范数及在原生读出、总位移上的投影。",
               f"前八个方向解释供体位移能量的 {100*energy:.2f}%，在受体上的平均覆盖为 {100*s['mean_heldout_energy']:.2f}%。",
               f"去除真实方向与去除匹配方向相比，双向成功减少量为 {fmt(s['drop_vs_geometry'])}（97.5%）。",
               f"仅保留真实方向相比保留匹配方向，双向成功增加量为 {fmt(s['keep_vs_geometry'])}（97.5%）。",
               "","## 直接移植第 17 层的查询诱发位移",""])
    x=load(ROOT/"READOUT_SUMMARY.json")
    raw=load(ROOT/"READOUT_RAW.json")["rows"]
    bank=load(ROOT/"DISJOINT_BASES.json")
    d.extend([f"从互斥百位供体学习的子空间覆盖 {sum(z['basis_ready'] for z in raw)}/{x['n']} 道受体。各百位分组的方向样本数为 {[bank[str(h)]['n'] for h in range(2,7)]}。",
              "", "| 干预 | K1 答对 | K2 恰好多 100 | 同题双向 | 等于跨数字供体正解或错值的输出 |",
              "|---|---:|---:|---:|---:|"])
    for m,v in x["outcomes"].items():
        d.append(f"| {m} | {v['K1_correct']}/{x['n']} | {v['K2_foil']}/{x['n']} | {v['joint']}/{x['n']} | {v['donor_candidate']} |")
    for k,v in x["comparisons"].items():
        d.extend(["",f"{k} 的双向净差为 {fmt(v)}（97.5%）。"])
    d.extend(["","同题结构复位、用于复现同题效应的低秩投影、从另一题直接供给校正位移是三种不同检验。前两者可以成立而第三种失败。数学定义见 METHOD.md，逐题记录与哈希核验见 results/。",""])
    Path("REPORT_ZH.md").write_text("\n".join(d),encoding="utf-8")


if __name__=="__main__":
    print(verify(),flush=True)
    render()

