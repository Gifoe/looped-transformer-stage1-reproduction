"""Rebuild every reported summary from frozen test inputs and actual decoded tokens."""
import json
from pathlib import Path
import platform

import torch
import transformers

import boundary_control
import causal
import natural
import pathway
import rule
from bidi import prior_pairs, unordered
from native import MODEL_HASH, REV, load, save, sha

ROOT=Path("results")
PILOT=Path("/root/ouro-open-carry-20261121/MANIFEST.json")
BOUNDARY=Path("/root/ouro-carry-boundary-20261121/results/MANIFEST.json")
PREVIOUS=Path("/root/ouro-bidirectional-20261121/results/DECK.json")


def all_text(rows):
    for u in rows:
        assert u["tokens"] and rule.parse(u["text"])==u["number"]


def compare(obj,ref):
    assert json.loads(json.dumps(obj))==ref


def base():
    manifest=load(ROOT/"DECK.json")
    raw=load(ROOT/"NATIVE.json")
    assert manifest["prefix"]==rule.PREFIX and raw["deck_sha256"]==sha(ROOT/"DECK.json")
    assert manifest["source_hashes"]==[sha(x) for x in (PILOT,BOUNDARY,PREVIOUS)]
    past=prior_pairs(PILOT,BOUNDARY)
    past.update(unordered(p) for row in load(PREVIOUS)["rows"] for p in row["pairs"])
    past.update(unordered(p) for p in rule.DEMONSTRATIONS)
    assert manifest["historical_unique_unordered"]==len(past)
    n=len(past)
    assert len(manifest["rows"])==len(raw["rows"])==216
    assert len({g["id"] for g in raw["rows"]})==216
    universe={}
    pools={}
    for x,y in zip(manifest["rows"],raw["rows"],strict=True):
        assert x=={k:v for k,v in y.items() if k!="native"}
        a0,a1=x["a"]
        b,b2=x["b"],x["source_b"]
        assert a1-a0==x["delta"] and 20<=abs(b-b2)<=59 and x["delta"]!=abs(b-b2)
        assert x["target"]==[[rule.value(a,b,o) for o in (0,1)] for a in (a0,a1)]
        for o,named in enumerate(("carry_flip","borrow_flip")):
            each=[boundary_control.needs(a,b,o) for a in (a0,a1)]
            assert x[named]==(each[0]!=each[1])
        pairs={unordered((a,q)) for a in x["a"] for q in (b,b2)}
        assert len(pairs)==4 and not pairs.intersection(past)
        past.update(pairs)
        pools.setdefault(x["split"],set()).update(x["a"])
        assert len(y["native"])==2 and all(len(q)==4 for q in y["native"])
        for outputs in y["native"]:
            all_text(outputs)
        universe[x["id"]]=y
    assert not pools["discovery"].intersection(pools["confirmation"])
    compare(rule.counts(raw["rows"]),load(ROOT/"NATIVE_SUMMARY.json"))
    return universe,past,len(past)-n


def verify():
    index,seen,n_new=base()
    s=load(ROOT/"NATIVE_SUMMARY.json")
    v=dict(status="PASS", model=REV,model_sha256=MODEL_HASH,
           python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
           cuda=torch.version.cuda,gpu="NVIDIA GeForce RTX 4090",
           math="BF16 checkpoint exactly promoted to FP32, eager native no cache, TF32 and autocast disabled",
           ab_pairs_original_and_donor_unique=n_new,
           deck_sha256=sha(ROOT/"DECK.json"),native_sha256=sha(ROOT/"NATIVE.json"))
    for diagnostic in ("DESCRIPTIVE.json","CARRY_DESCRIPTIVE.json"):
        assert load(ROOT/diagnostic)["native_sha256"]==v["native_sha256"]
    if (ROOT/"GRAFT.json").exists():
        r=load(ROOT/"GRAFT.json")
        assert r["native_sha256"]==v["native_sha256"] and s["confirmation"]["gate"]
        assert [x["id"] for x in r["rows"]]==list(index)
        a=load(ROOT/"GRAFT_SUMMARY.json")
        compare(causal.summary(r["rows"]),a)
        by_result={}
        for x in r["rows"]:
            g=index[x["id"]]
            assert g["split"]==x["split"] and g["band"]==x["band"]
            by_result[x["id"]]=x
            if not x["aligned"]:
                continue
            assert len(x["cases"])==2
            for j,c in enumerate(x["cases"]):
                assert c["a"]==g["a"][j] and c["alt"]==g["a"][1-j] and c["b"]==g["b"]
                assert c["b2"]==g["source_b"] and c["original"]==g["target"][j]
                assert c["new"]==[rule.value(c["alt"],c["b"],o) for o in (0,1)]
                assert c["source"]==[rule.value(c["alt"],c["b2"],o) for o in (0,1)]
                assert all(len(set(t))==3 for t in zip(c["original"],c["new"],c["source"],strict=True))
                assert c["positions"]==list(range(c["positions"][0],c["positions"][-1]+1))
                for mode in causal.MODES:
                    all_text(x[mode][j])
                assert x["native"][j]==g["native"][1][j*2:j*2+2]
        v["graft_sha256"]=sha(ROOT/"GRAFT.json")
        if (ROOT/"PATH_RAW.json").exists():
            p=load(ROOT/"PATH_RAW.json")
            select=load(ROOT/"SELECTION.json")
            assert select["primary_raw_sha256"]==p["primary_raw_sha256"]==v["graft_sha256"]
            assert p["selection_sha256"]==sha(ROOT/"SELECTION.json") and a["confirmation"]["gate"]
            assert len({y["id"] for y in p["rows"]})==len(p["rows"])
            assert [(r["id"],r["orientation"]) for r in p["rows"]]==[
                (x["id"],k) for x,_,k in pathway.pick(ROOT,"confirmation")]
            lookup=pathway.pick(ROOT,"discovery")
            assert select["n_discovery"]==len(lookup)
            assert all(x["split"]=="discovery" for x,_,_ in lookup)
            for y in p["rows"]:
                x=by_result[y["id"]]
                k=y["orientation"]
                assert x["split"]=="confirmation" and y["case"]==x["cases"][k]
                for mode in ("native","graft"):
                    assert x[mode][k]==y["outputs"][mode]
                for gen in y["outputs"].values():
                    all_text(gen)
            compare(pathway.summarize(p["rows"]),load(ROOT/"PATH_SUMMARY.json"))
            v["path_sha256"]=sha(ROOT/"PATH_RAW.json")
    if (ROOT/"NATURAL_RAW.json").exists():
        y=load(ROOT/"NATURAL_RAW.json")
        assert y["native_sha256"]==v["native_sha256"]
        s=load(ROOT/"NATURAL_SUMMARY.json")
        selected,eligible=natural.pick(list(index.values()))
        assert [(q["cohort"],q["id"],q["orientation"],q["rule"]) for q in y["rows"]]==[
            (z,g["id"],i,o) for z,g,i,o in selected]
        assert eligible==s["eligible_before_cap"]
        compare(dict(natural.summarize(y["rows"]),eligible_before_cap=s["eligible_before_cap"]),s)
        for z in y["rows"]:
            g=index[z["id"]]
            j=2*z["orientation"]+z["rule"]
            assert z["target"]==g["target"][z["orientation"]][z["rule"]]
            p,q=[g["native"][k][j] for k in (0,1)]
            assert q["number"]==z["target"]
            if z["cohort"]=="repair":
                assert p["number"]==z["wrong"]!=z["target"]
            else:
                assert p["number"]==z["target"]
            if z["aligned"]:
                assert z["outputs"]["native"]==q
                all_text(z["outputs"].values())
        v["natural_sha256"]=sha(ROOT/"NATURAL_RAW.json")
    a=load(ROOT/"CONTROL_DECK.json")
    b=load(ROOT/"CONTROL_RAW.json")
    assert b["manifest_sha256"]==sha(ROOT/"CONTROL_DECK.json")
    assert a["source_sha256"]==[sha(x) for x in (PILOT,BOUNDARY,PREVIOUS,ROOT/"DECK.json")]
    assert len(a["rows"])==len(b["rows"])==256 and len({x["id"] for x in a["rows"]})==256
    for d,f in zip(a["rows"],b["rows"],strict=True):
        assert d=={k:u for k,u in f.items() if k!="pred"}
        for k,(a1,b1) in enumerate(d["pairs"]):
            assert unordered((a1,b1)) not in seen
            seen.add(unordered((a1,b1)))
            assert rule.value(a1,b1,d["op"])==d["target"]
            assert boundary_control.needs(a1,b1,d["op"])==bool(k)
        for preds in f["pred"]:
            all_text(preds)
    compare(boundary_control.summary(b["rows"]),load(ROOT/"CONTROL_SUMMARY.json"))
    v["same_answer_holdout_sha256"]=sha(ROOT/"CONTROL_RAW.json")
    save(ROOT/"VERIFICATION.json",v)
    return v


def pp(e,scale=1.0):
    a,b=e["interval"]
    return f"{100*e['mean']/scale:+.2f} [{100*a/scale:+.2f}, {100*b/scale:+.2f}]"


def render():
    v=load(ROOT/"NATIVE_SUMMARY.json")["confirmation"]
    desc=load(ROOT/"DESCRIPTIVE.json")["results"]["confirmation"]["counts"]
    control=load(ROOT/"CONTROL_SUMMARY.json")
    d=["# Ouro-1.4B Base 如何随算术规则使用跨循环操作数","",
       "运行固定检查点与原生共享层，以完整词表贪心生成第一个非空行的整数。发现与确认的首操作数数值池分离。所有阈值、对照与原始 token 序列随实验发布。","",
       "## 同一输入模板下的原生变化","",
       "| 规则 | 原生 K1 正确 | 原生 K2 正确 | K1→K2 自然纠正 | K1 错答恰是另一规则的正确值 |",
       "|---|---:|---:|---:|---:|"]
    for i,name in enumerate(("加法 A+B","反向减法 B−A")):
        d.append(f"| {name} | {v[f'K1_rule{i}_correct']}/{v['n_rule']} | {v[f'K2_rule{i}_correct']}/{v['n_rule']} | {desc[f'rule{i}_repair']} | {desc.get(f'rule{i}_wrong_rule_integer',0)} |")
    d.extend(["",f"{v['K2_all_four']}/{v['n_stems']} 个确认题干的四道题在 K2 都正确。自然修复有 {v['K1_to_K2_repair']} 道，回退 {v['K1_to_K2_damage']} 道。","",
              "## 同一份 H1 首操作数状态在两条规则中的作用",""])
    if (ROOT/"GRAFT_SUMMARY.json").exists():
        w=load(ROOT/"GRAFT_SUMMARY.json")["confirmation"]
        g=w["graft"]
        n=w["noise"]
        ind=w["native"]
        d.extend(["在两种规则中只移植相同的早期首操作数状态，保持 H1 答案读出不变。若两题都生成相应反事实整数，增加首操作数应让加法答案增加、反向减法答案减少。规则无关的固定方向移位因此可用这组对照检验。","",
                  "| 条件 | 加法准确反事实 | 反向减法准确反事实 | 同一方向的两规则都准确 | 至少一个方向成功的题干 | 两个移植方向都成功的题干 |",
                  "|---|---:|---:|---:|---:|---:|"])
        for name,z in (("原生复跑",ind),("自然 H1 移植",g),("等范数和角度对照",n)):
            d.append(f"| {name} | {z['per_rule'][0]}/{z['orientations']} | {z['per_rule'][1]}/{z['orientations']} | {sum(z['per_orientation'])}/{z['orientations']} | {z['stems_joint']}/{v['n_stems']} | {z['both_orientations']}/{v['n_stems']} |")
        d.extend(["",f"对齐 {g['alignable']}/{v['n_stems']} 组。相对匹配对照的两规则准确率增量为 {pp(w['excess_pairs_per_stem'],2)} 个百分点（95% 区间），预设门槛{'通过' if w['gate'] else '未通过'}。",
                  f"移植中符合两条规则同加该差值与两条规则同减该差值的次数分别为 {g['fixed_plus']} 和 {g['fixed_minus']}。两规则精确生成供体整题答案 {g['donor_integer']} 次。三种移植幅度的两规则正确数为 {g['bands']}，分母 {g['band_n']}。进位状态翻转有 {g['carry_joint']}/{g['carry_n']} 次两规则都准确，借位翻转有 {g['borrow_joint']}/{g['borrow_n']} 次。原生纠错子集移植命中数 {g['repair_matches']}，分母 {g['repair_base']}。"])
    else:
        d.append("原生基线未通过，按协议跳过本项。")
    if (ROOT/"GRAFT.json").exists():
        cases=load(ROOT/"GRAFT.json")["rows"]
        example=next((x for x in cases if x["split"]=="confirmation" and x["aligned"] and
                      x["carry_flip"] and x["borrow_flip"] and
                      all([r["number"] for r in left]==c["original"] and
                          [r["number"] for r in right]==c["new"]
                          for left,right,c in zip(x["native"],x["graft"],x["cases"],strict=True))),None)
        if example:
            c=example["cases"][0]
            d.extend(["",f"例如 {example['id']} 保持 B={c['b']}，把首操作数状态从 {c['a']} 换成 {c['alt']}。实际自由生成的加法由 {c['original'][0]} 变为 {c['new'][0]}，反向减法由 {c['original'][1]} 变为 {c['new'][1]}。该对跨过两种百位传递边界。"])
    if (ROOT/"PATH_SUMMARY.json").exists():
        p=load(ROOT/"PATH_SUMMARY.json")
        z=load(ROOT/"SELECTION.json")
        sites=lambda label:[i+1 for i in (z[label] or [])]
        d.extend(["","## 第二轮注意力 V 路径","",
                  f"发现集有 {z['n_discovery']} 个满足条件的题干，锁定的层窗口为 {sites('window')}，单层为 {sites('layer')}。确认路径样本只纳入原生正确且自然移植已使两规则正确反事实的题干。确认样本数 {p['n']}。","",
                  "| 处理 | 两规则恢复原答案 | 两规则仍生成反事实 |",
                  "|---|---:|---:|"])
        for arm,m in p["modes"].items():
            d.append(f"| {arm} | {m['joint_original']}/{p['n']} | {m['joint_counterfactual']}/{p['n']} |")
        for arm,e in p["paired"].items():
            d.append(f"\n{arm} 双规则精确恢复相对同位置几何对照的增量为 {pp(e)} 个百分点。全层区间为 95%，选中窗口和单层的区间为 97.5%。")
    if (ROOT/"NATURAL_SUMMARY.json").exists():
        n=load(ROOT/"NATURAL_SUMMARY.json")
        r=n["repair"]
        q=n["control"]
        d.extend(["","## 回拨首操作数位置的原生 K1 V","",
                  f"共有 {n['eligible_before_cap']} 道具备完整错误整数的原生修复题。抽取 {r['n']} 道（{r['n_stems']} 题干），另有 {q['n']} 道原生两轮均正确的匹配对照。","",
                  "| 处理 | 修复题精确重现 K1 错答 | 修复题仍正确 | 对照题仍正确 |",
                  "|---|---:|---:|---:|"])
        for arm,w in r["arms"].items():
            d.append(f"| {arm} | {w.get('exact_K1',0)}/{r['aligned']} | {w.get('original',0)}/{r['aligned']} | {q['arms'].get(arm,{}).get('original',0)}/{q['aligned']} |")
        if (ROOT/"ROLLBACK_DESCRIPTIVE.json").exists():
            d.extend(["","按原生错答是否恰为漏进位或借位的细分结果见 results/ROLLBACK_DESCRIPTIVE.json。"])
        if (e:=r["matched_excess"].get("all")):
            d.append(f"\n全部层回拨相对匹配对照的精确重现超额为 {e['total']} 答案，题干 bootstrap 区间为 {e['interval']} 次每题干。原生纠错路径门槛{'通过' if n['gate'] else '未通过'}。")
    d.extend(["","## 独立同正确答案的百位进借位确认","",
              "| 规则 | 条件 | K1 正确 | K2 正确 | 指定 ±100 错答后精确修复 |",
              "|---|---|---:|---:|---:|"])
    for o,name in enumerate(("加法","反向减法")):
        q=control[str(o)]
        for k,label in (("need","需要向百位传递"),("no_need","无需传递")):
            w=q[k]
            d.append(f"| {name} | {label} | {w['K1_correct']}/{q['n']} | {w['K2_correct']}/{q['n']} | {w['foil_repair']}/{q['n']} |")
    for o,name in enumerate(("加法","反向减法")):
        row=control[str(o)]
        d.extend(["",f"{name}配对超额为 {pp(row['paired'])} 个百分点（97.5% 区间），门槛{'通过' if row['gate'] else '未通过'}。"])
    d.extend(["","同答案配对控制了正确整数和精确错误整数的词元表面形式。修复签名属于行为证据。状态移植、选择后的 V 路径和原生回拨的证据不能互相替代。规则感知的候选更新与重新计算可以产生相同最终整数。","",
              "各结果文件的 SHA256、配对结构及从逐题序列到汇总的验证见 results/VERIFICATION.json。",""])
    Path("REPORT_ZH.md").write_text("\n".join(d),encoding="utf-8")


if __name__=="__main__":
    print(verify(),flush=True)
    render()

