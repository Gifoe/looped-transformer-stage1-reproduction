"""Validate complete token records and render scope-specific experimental findings."""
from collections import Counter
import json
from pathlib import Path
import platform

import torch
import transformers

import borrow
import context
import interchange
import routing
import suffix
import transfer
from bidi import unordered
from native import MODEL_HASH, REV, load, save, sha
from rule import PREFIX, parse

ROOT=Path("results")


def checks(x):
    assert x["tokens"] and parse(x["text"])==x["number"]


def same(l,r):
    assert json.loads(json.dumps(l))==r


def verify():
    deck=load(ROOT/"DECK.json")
    raw=load(ROOT/"NATIVE.json")
    assert deck["source_sha256"]==[sha(q) for q in borrow.INPUTS]
    assert deck["prompt"]==PREFIX and raw["deck_sha256"]==sha(ROOT/"DECK.json")
    assert len(deck["rows"])==len(raw["rows"])==396
    seen=borrow.history()
    assert len(seen)==deck["historical_unordered"]
    index={}
    groups=Counter()
    for a,b in zip(deck["rows"],raw["rows"],strict=True):
        assert a=={k:v for k,v in b.items() if k!="pred"}
        t=b["target"]
        assert deck["targets"][str(t)]==b["split"] and b["foil"]==t+100
        assert b["id"] not in index
        index[b["id"]]=b
        groups[b["split"],t]+=1
        for j,(x,y) in enumerate(b["pairs"]):
            assert unordered((x,y)) not in seen and y-x==t and borrow.needs(x,y)==bool(j)
            seen.add(unordered((x,y)))
        assert len(b["pred"])==2 and all(len(z)==2 for z in b["pred"])
        for z in b["pred"]:
            for q in z:
                checks(q)
    assert len(groups)==150 and all(n==(2 if k=="discovery" else 3) for (k,_),n in groups.items())
    baseline=load(ROOT/"NATIVE_SUMMARY.json")
    same(borrow.report(raw["rows"],deck["targets"]),baseline)
    v=dict(status="PASS", model=REV,model_sha256=MODEL_HASH,
           python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
           gpu="NVIDIA GeForce RTX 4090",precision="FP32 from pinned BF16 checkpoint; native no cache; no autocast or TF32",
           deck_sha256=sha(ROOT/"DECK.json"),native_sha256=sha(ROOT/"NATIVE.json"),
           unique_new_operand_pairs=len(seen)-deck["historical_unordered"],independent_target_bins=len(groups))
    if baseline["gate"]:
        qpath=ROOT/"QUERY_RAW.json"
        selection=load(ROOT/"SELECTION.json")
        query=load(qpath)
        assert selection["native_sha256"]==query["native_sha256"]==v["native_sha256"]
        assert query["selection_sha256"]==sha(ROOT/"SELECTION.json")
        assert len(selection["layer_scores"])==72 and len(selection["head_scores"])==16
        assert selection["layer"]==max(selection["layer_scores"][:24],key=lambda x:x["rank"])["layer"]
        assert selection["head"]==max(selection["head_scores"],key=lambda x:x["rank"])["head"]
        assert selection["write"]==max(selection["layer_scores"][24:],key=lambda x:x["rank"])
        for row in selection["by_case"]+selection["head_by_case"]:
            assert index[row["id"]]["split"]=="discovery" and borrow.native_event(index[row["id"]],1)
        fixes=[x for x in index.values() if x["split"]=="confirmation" and borrow.native_event(x,1)]
        ids=[(x["id"],k) for x in sorted(fixes,key=lambda row:row["id"]) for k in ("repair","safe")]
        assert ids==[(x["id"],x["kind"]) for x in query["rows"]]
        for q in query["rows"]:
            original=index[q["id"]]
            assert q["target"]==original["target"] and q["foil"]==original["foil"]
            expected=[z[q["kind"]=="repair"] for z in original["pred"]]
            assert q["pred"]["clean"]==q["pred"]["head_graphcut"]==expected
            assert set(q["pred"])==set(interchange.MODES)
            for arm,x in q["pred"].items():
                for j,c in enumerate(x):
                    checks(c)
                    count=len(q["prefix"])
                    if not q["reach"][j]:
                        assert c==expected[j]
                    else:
                        assert c["tokens"][:count]==q["prefix"]
                        assert c["tokens"][count]==q["pivot_token"][arm][j]
        same(interchange.summarize(query["rows"]),load(ROOT/"QUERY_SUMMARY.json"))
        v.update(query_sha256=sha(qpath),query_selection_sha256=sha(ROOT/"SELECTION.json"))
        cp=ROOT/"CONTEXT_RAW.json"
        next_=load(cp)
        assert next_["query_sha256"]==v["query_sha256"]
        cs=load(ROOT/"CONTEXT_SELECTION.json")
        assert next_["selection_sha256"]==sha(ROOT/"CONTEXT_SELECTION.json") and cs["source_sha256"]==v["native_sha256"]
        assert len(query["rows"])==len(next_["rows"])
        for a,b in zip(query["rows"],next_["rows"],strict=True):
            assert all(a[k]==b[k] for k in ("id","kind","target","foil"))
            assert a["pred"]["clean"]==b["pred"]["clean"]
            for p in b["pred"].values():
                for z in p:
                    checks(z)
        same(context.summarize(next_["rows"]),load(ROOT/"CONTEXT_SUMMARY.json"))
        v.update(context_sha256=sha(cp),context_selection_sha256=sha(ROOT/"CONTEXT_SELECTION.json"))
        routes=ROOT/"ROUTING_RAW.json"
        if routes.exists():
            d=load(routes)
            opt=load(ROOT/"ROUTING_SELECTION.json")
            assert d["selection_sha256"]==sha(ROOT/"ROUTING_SELECTION.json") and d["query_sha256"]==v["query_sha256"]
            assert opt["layer"]==selection["layer"] and opt["query_selection_sha256"]==query["selection_sha256"]
            assert all(index[r["id"]]["split"]=="discovery" for r in opt["items"])
            assert len(d["rows"])==len(query["rows"])
            for a,b in zip(d["rows"],query["rows"],strict=True):
                assert all(a[k]==b[k] for k in ("id","kind","target","foil"))
                assert a["pred"]["replay"]==b["pred"]["layer"] and a["native"]==b["pred"]["clean"]
                for m in a["pred"].values():
                    for z in m:
                        checks(z)
            same(routing.summarize(d["rows"]),load(ROOT/"ROUTING_SUMMARY.json"))
            v["routing_sha256"]=sha(routes)
        tail=ROOT/"SUFFIX_RAW.json"
        if tail.exists():
            data=load(tail)
            assert data["source_sha256"]==v["context_sha256"] and data["layer"]==cs["pick"]["readout"]
            fixtures=[r for r in next_["rows"] if r["kind"]=="repair"]
            for a,b in zip(data["rows"],fixtures,strict=True):
                assert all(a[k]==b[k] for k in ("id","target","foil"))
                assert a["outputs"]["native_suffix"]==b["pred"]["full"] and a["outputs"]["full"]==b["pred"]["clean"][::-1]
                for outputs in a["outputs"].values():
                    for z in outputs:
                        checks(z)
            same(suffix.summarize(data["rows"]),load(ROOT/"SUFFIX_SUMMARY.json"))
            v["suffix_sha256"]=sha(tail)
        tf=ROOT/"TRANSFER_RAW.json"
        if tf.exists():
            t=load(tf)
            assert t["query_sha256"]==v["query_sha256"] and t["selection_sha256"]==query["selection_sha256"]
            for a in t["rows"]:
                if a["matched"]:
                    assert index[a["donor"]]["split"]=="discovery"
                    assert {a["target"]//100,a["foil"]//100}.isdisjoint({a["donor_target"]//100,(a["donor_target"]+100)//100})
                    for m in a["pred"].values():
                        for z in m:
                            checks(z)
            same(transfer.summarize(t["rows"]),load(ROOT/"TRANSFER_SUMMARY.json"))
            v["transfer_sha256"]=sha(tf)
    save(ROOT/"VERIFICATION.json",v)
    return v


def pp(e,scale=1):
    a,b=e["interval"]
    return f"{100*e['mean']/scale:+.2f} [{100*a/scale:+.2f}, {100*b/scale:+.2f}]"


def counts(e):
    return f"{e['total']:+d}，按目标聚类的每目标净增量 {e['mean']:+.3f}，区间 [{e['interval'][0]:+.3f}, {e['interval'][1]:+.3f}]"


def render():
    native=load(ROOT/"NATIVE_SUMMARY.json")
    b=native["confirmation"]
    raw=load(ROOT/"NATIVE.json")
    n=288
    d=["# Ouro-1.4B Base 反向减法的自然漏借位与跨轮查询机制","",
       "保留上一轮固定提示和冻结检查点。逐题以完整词表贪心生成第一个非空行的整数。每个条件的匹配题具有完全同一个正确答案和 +100 错值。发现集与确认集的正确整数及抽样首操作数池分离。","",
       "| 确认条件 | K1 正确 | K2 正确 | K1 恰多100→K2 完整正确 |",
       "|---|---:|---:|---:|"]
    for k,label in (("borrow","需要向百位借位"),("safe","无需向百位借位")):
        x=b[k]
        d.append(f"| {label} | {x['K1_correct']}/{x['n']} | {x['K2_correct']}/{x['n']} | {x['exact_repair']}/{x['n']} |")
    e=b["paired_excess_per_target"]
    d.extend(["",f"确认组覆盖 {b['unique_borrow_repairs']} 个出现精确修复的不同结果整数。每个目标有三个题对。配对差为 {pp(e,3)} 个百分点（95% 区间）。行为门槛{'通过' if native['gate'] else '未通过'}。",""])
    if not native["gate"]:
        d.append("按锁定协议，未开展干预。")
        Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")
        return
    s=load(ROOT/"SELECTION.json")
    q=load(ROOT/"QUERY_SUMMARY.json")
    r=q["repair"]
    safe=q["safe"]
    head=s["head"]
    l=s["layer"]
    bst=s["head_scores"][head]
    w=s["write"]
    d.extend(["## 查询和残差写入的双向干预","",
              f"发现集共 {s['all_native_repairs']} 个精确修复，{s['n_discovery']} 个前缀对齐。选中的 Q 为物理第 {l+1} 层、第 {head+1} 个头（从 1 计数）。两向首差异词元的平均诊断增益为 {bst['mean_directional_nats'][0]:.3f}、{bst['mean_directional_nats'][1]:.3f} logits。另锁定第 {w['layer']+1} 层的 {w['site']}。逐例分数在 SELECTION.json。",
              f"确认集有 {r['n']} 道自然修复题及 {safe['n']} 道同答案安全对照。双轮都自然到达锁定前缀的题分别有 {r['both_reach']} 和 {safe['both_reach']}。只有被实际改变的枢轴词元会进入新的完整词表续写。","",
              "| 干预 | K1 恢复完整正解 | K2 重现同一个 +100 错答 | 同题双向 | 对照题新损伤 K1/K2 |",
              "|---|---:|---:|---:|---:|"])
    for name,a in r["modes"].items():
        c=safe["modes"][name]
        d.append(f"| {name} | {a['K1_correct']}/{r['n']} | {a['K2_foil']}/{r['n']} | {a['joint']}/{r['n']} | {c['wrong_created_K1']}/{c['wrong_created_K2']} |")
    d.extend(["",f"单头主要门槛{'通过' if q['primary_gate'] else '未通过'}。按目标聚类的两个主要对照与全部次级对照见 QUERY_SUMMARY.json。"])
    for key,a in q["comparisons"].items():
        d.extend(["",f"{key} 的双向净增量 {counts(a['joint'])}。"])
    c=load(ROOT/"CONTEXT_SUMMARY.json")
    ct=load(ROOT/"CONTEXT_SELECTION.json")
    d.extend(["","## 保持读出或保留上下文的层输入交换","",
              f"更换读出的位置在第 {ct['pick']['readout']+1} 层输入，更换早期上下文的位置在第 {ct['pick']['context']+1} 层输入。位置由发现集选择。整个输入替换在枢轴词元步骤与供体自然轨迹逐位一致，其后的所有整数仍需自由续写。","",
              "| 操作 | K1 恢复完整正解 | K2 再现准确 +100 | 同题双向 | 对照新增损伤 K1/K2 |",
              "|---|---:|---:|---:|---:|"])
    t=c["repair"]
    controls=c["safe"]
    for a,y in t["arms"].items():
        z=controls["arms"][a]
        d.append(f"| {a} | {y['K1_correct']}/{t['n']} | {y['K2_foil']}/{t['n']} | {y['joint']}/{t['n']} | {z['K1_harm']}/{z['K2_harm']} |")
    for a in context.PARTS:
        d.extend(["",f"{a} 相对同位置几何对照的双向净增量 {counts(c[a+'_excess'])}（97.5% 区间），预设门槛{'通过' if c[a+'_gate'] else '未通过'}。"])
    output_r=ROOT/"ROUTING_SUMMARY.json"
    if output_r.exists():
        rr=load(output_r)
        selected=load(ROOT/"ROUTING_SELECTION.json")
        field=routing.FIELDS[selected["field"]]
        panel=rr["repair"]
        d.extend(["","## 第 14 层查询效应的来源分解","",
                  f"发现集锁定的区域为 {field}。这项分析复用查询确认集，属于同一队列里的机制分解。",
                  "", "| 干预 | K1 正解 | K2 精确 +100 | 同题双向 |", "|---|---:|---:|---:|"])
        for arm,j in panel["arms"].items():
            d.append(f"| {arm} | {j['K1_correct']}/{panel['n']} | {j['K2_foil']}/{panel['n']} | {j['joint']}/{panel['n']} |")
        d.extend(["","区域贡献按固定 V 的线性和计算。仅保留一个区域或去掉该区域时，总权重可以不为 1。between 和 within 保持合法归一化注意力。"])
    alt=ROOT/"SUFFIX_SUMMARY.json"
    if alt.exists():
        z=load(alt)
        n=c["repair"]["n"]
        d.extend(["","## 实际枢轴 token 之后的自由续写","",
                  "复用整层输入枢轴干预实际产生的首个变化 token。随后的供体状态均在每条模型分支已经生成的前缀上重新计算。全状态持续复放复现对侧原始答案，是结构校验。","",
                  "| 续写方式 | K1 正解 | K2 重现+100 | 同题双向 | 枢轴更换后仍失败的 K1 子集里恢复 |",
                  "|---|---:|---:|---:|---:|"])
        for name,m in z.items():
            d.append(f"| {name} | {m['K1_correct']}/{n} | {m['K2_foil']}/{n} | {m['joint']}/{n} | {m['K1_correct_after_failed_pivot']}/{m['failed_pivot']} |")
    d.extend(["","写入分支预先定义的跨样本可迁移实验"+("已执行，见 TRANSFER_SUMMARY.json。" if (ROOT/"TRANSFER_SUMMARY.json").exists() else "未达到启动条件。")])
    d.extend(["","单层 Q、读出和上下文干预分别回答不同的物理路径问题。同一样本内的枢轴词元效应与完整整数切换单独记录。全输入回放是正确性检查。读出状态可以含有已经形成的候选。未经干预的原生模型权重在每项实验中保持冻结。","",
              "数据来源、逐题文本、每个生成词元、匹配与重算验证和源码均在本目录。"])
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__=="__main__":
    print(verify(),flush=True)
    render()

