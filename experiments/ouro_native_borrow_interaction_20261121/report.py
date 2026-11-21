"""Independently verify native data, exact outcomes, matching, and measured claims."""
import json
from pathlib import Path
import platform

from PIL import Image
import torch
import transformers

from bidi import unordered
from borrow import needs
import mediation
import native_deck
import readout
import specificity
import trajectory
from native import MODEL_HASH, REV, load, save, sha
from rule import PREFIX, parse

ROOT=Path("results")
MODEL="/root/autodl-tmp/ouro-cross-task-20261118/model"


def equal(a,b):
    assert json.loads(json.dumps(a))==b


def parsed(q):
    assert q["tokens"] and q["number"]==parse(q["text"])


def verify():
    m=load(ROOT/"DECK.json")
    actual=load(ROOT/"NATIVE.json")
    assert m["prompt"]==PREFIX and m["previous_sha256"]==sha(native_deck.EXTRA)
    assert actual["deck_sha256"]==sha(ROOT/"DECK.json") and len(actual["rows"])==len(m["rows"])==280
    past=native_deck.history()
    assert len(past)==m["earlier_unordered"]
    start=len(past)
    pool={}
    for a,x in zip(m["rows"],actual["rows"],strict=True):
        assert a=={k:v for k,v in x.items() if k!="pred"}
        assert a["id"] not in pool
        pool[a["id"]]=x
        target=a["target"]
        assert (target+100)==a["foil"] and a["split"]==m["groups"][str(target)]
        for arm in ("borrow","safe"):
            pair=a[arm]
            a0,b0=pair[0]
            a1,b1=pair[3]
            assert pair[1]==[a1,b0] and pair[2]==[a0,b1]
            assert a0//100==a1//100 and b0//100==b1//100
            assert [a1-a0,b0-b1]==a[arm+"_steps"]
            if arm=="borrow":
                assert [needs(*p) for p in pair]==[False,False,False,True]
                assert b0-a0==target+100
            else:
                assert not any(needs(*p) for p in pair)
            assert b1-a1==target
            for p in pair:
                y=unordered(p)
                assert y not in past
                past.add(y)
        assert len(x["pred"])==2 and all(len(z)==5 for z in x["pred"])
        for u in x["pred"]:
            for z in u:
                parsed(z)
    assert len(past)-start==2240
    a=native_deck.report(actual["rows"],m["groups"])
    equal(a,load(ROOT/"NATIVE_SUMMARY.json"))
    status=dict(status="PASS", model_revision=REV, model_sha256=MODEL_HASH,
                python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
                cuda=torch.version.cuda,gpu="NVIDIA GeForce RTX 4090",
                math="BF16 checkpoint promoted to FP32, eager no cache, TF32 and autocast off",
                unique_quartet_prompts=len(past)-start,scored_native_prompts=1400,
                native_sha256=sha(ROOT/"NATIVE.json"),deck_sha256=sha(ROOT/"DECK.json"))
    if not a["gate"]:
        save(ROOT/"VERIFICATION.json",status)
        return
    tok=transformers.AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True)
    chosen,evaluate,matches=readout.plan(tok,actual["rows"])
    b=load(ROOT/"MATCHING.json")
    assert b["native_sha256"]==status["native_sha256"]
    assert b["bank"]==[z["g"]["id"] for z in chosen]
    assert b["targets"]==[z["g"]["id"] for z in evaluate]
    assert b["matches"]==dict(zip(b["targets"],matches,strict=True))
    for id,(d,near) in b["matches"].items():
        if d:
            assert abs(pool[id]["target"]//100-pool[d]["target"]//100)>=2
            assert pool[id]["split"]=="confirmation" and pool[d]["split"]=="discovery"
        if near:
            assert pool[id]["target"]!=pool[near]["target"] and pool[id]["target"]//100==pool[near]["target"]//100
    c=load(ROOT/"READOUT_RAW.json")
    assert c["native_sha256"]==status["native_sha256"] and c["matching_sha256"]==sha(ROOT/"MATCHING.json")
    ordered=[(v["aligned"],q) for v in evaluate for q in v["pair"]]
    for row,(valid,case) in zip(c["rows"],ordered,strict=True):
        assert row["aligned"]==valid and all(row[k]==case[k] for k in ("id","kind","target","foil","prefix","reach"))
        assert row["pred"]["native"]==case["native"]
        assert set(row["pred"])==set(readout.MODES)
        for values in row["pred"].values():
            for z in values:
                parsed(z)
    equal(readout.summary(c["rows"],b["matches"]),load(ROOT/"READOUT_SUMMARY.json"))
    d=load(ROOT/"PATH_RAW.json")
    assert d["native_sha256"]==c["native_sha256"] and d["matching_sha256"]==c["matching_sha256"]
    for r,s in zip(d["rows"],c["rows"],strict=True):
        assert all(r[k]==s[k] for k in ("id","kind","target","foil","aligned"))
        assert r["pred"]["native"]==s["pred"]["native"][1]==r["pred"]["full"]
        for z in r["pred"].values():
            parsed(z)
    equal(mediation.summary(d["rows"]),load(ROOT/"PATH_SUMMARY.json"))
    t=load(ROOT/"TRAJECTORY_RAW.json")
    eligible=[r for r in c["rows"] if r["aligned"]]
    assert t["native_sha256"]==status["native_sha256"]
    for r,s in zip(t["rows"],eligible,strict=True):
        assert (r["id"],r["kind"])==(s["id"],s["kind"]) and len(r["norm"])==3
        assert all(len(a)==24 for a in r["norm"]) and r["norm"][0][0]==0
    equal(trajectory.stats(t["rows"]),load(ROOT/"TRAJECTORY_SUMMARY.json"))
    with Image.open(ROOT/"TRAJECTORY.png") as im:
        assert im.format=="PNG" and im.width>=1000 and im.height>=450
        im.verify()
    s=load(ROOT/"SPECIFICITY_RAW.json")
    assert s["native_sha256"]==status["native_sha256"] and s["readout_sha256"]==sha(ROOT/"READOUT_RAW.json")
    for p,r in zip(s["rows"],eligible,strict=True):
        assert all(p[k]==r[k] for k in ("id","kind","target","foil"))
        assert p["pred"]["native"]==r["pred"]["native"]
        if p["kind"]=="repair":
            assert p["pred"]["same_vector"]==r["pred"]["delta"]
        for pair in p["pred"].values():
            for z in pair:
                parsed(z)
    equal(specificity.summary(s["rows"]),load(ROOT/"SPECIFICITY_SUMMARY.json"))
    for k in ("MATCHING.json","READOUT_RAW.json","PATH_RAW.json","TRAJECTORY_RAW.json",
              "TRAJECTORY.png","SPECIFICITY_RAW.json"):
        status[k.lower().replace(".","_")+"_sha256"]=sha(ROOT/k)
    save(ROOT/"VERIFICATION.json",status)
    return status


def paired(c):
    if not c["interval"]:
        return "无可计算配对区间"
    lo,hi=c["interval"]
    return f"{c['total']:+d} 例净差，每目标 {c['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def render():
    a=load(ROOT/"NATIVE_SUMMARY.json")
    z=a["confirmation"]
    nr=z["total"]["joint_repairs"]
    d=["# Ouro-1.4B 原生漏借位中的跨轮交互和路径复位","",
       "冻结模型和固定提示，使用实际完整整数和真实生成词元。120 个发现题干和 160 个确认题干的正确联合整数相互分离。每个借位题干的联合角点与安全题干共享完整正确答案。",
       "", "| 确认条件 | 原生 K1 完整正确 | 原生 K2 完整正确 |","|---|---:|---:|"]
    for p,u in z["per"].items():
        d.append(f"| {p} | {u['K1_correct']}/{z['n']} | {u['K2_correct']}/{z['n']} |")
    d.extend(["",f"自然精确修复为 {nr}/{z['n']}，同答案无需借位组为 {z['total']['matched_safe_repairs']}/{z['n']}。在这些精确修复中，{z['total']['other_three_K1_correct_among_joint_repairs']} 道的其他三个角点在 K1 已经全对。配对 {paired(z['paired'])}（95%区间）。"])
    if not a["gate"]:
        Path("REPORT_ZH.md").write_text("\n".join(d+["","预设原生行为门槛未通过。"])+"\n",encoding="utf-8")
        return
    ident=load(ROOT/"MATCHING.json")
    rr=load(ROOT/"READOUT_SUMMARY.json")
    n=rr["repair"]["n"]
    dis=sum(z[0] is not None for z in ident["matches"].values())
    aligned=sum(z[1] is not None for z in ident["matches"].values())
    d.extend(["","## 对第17层原生四角跨轮差分做干预","",
              f"对齐 {rr['repair']['aligned']}/{n}。可用供体 {len(ident['bank'])} 个，正误百位完全互斥供体匹配 {dis}/{n}，同百位不同完整答案匹配 {aligned}/{n}。干预是首个自然分歧位置之前的读出状态改动，其后按同一深度续写。",
              "", "| 干预 | K1 正解 | K2 恢复原 +100 | 同题双向 | 其余三角 K1 全正的双向 | 受体输出等于异数字供体或供体错值 K1/K2 |",
              "|---|---:|---:|---:|---:|---:|"])
    for p,v in rr["repair"]["modes"].items():
        d.append(f"| {p} | {v['K1_exact']}/{n} | {v['K2_foil']}/{n} | {v['joint']}/{n} | {v['tight_joint']}/{rr['repair']['tight']} | {v['K1_donor_collision']}/{v['K2_donor_collision']} |")
    for c,results in rr["comparisons"].items():
        d.extend(["",f"{c} 的同题双向配对差为 {paired(results['joint'])}（97.5%区间），两个单方向结果见原始汇总。"])
    d.extend(["",f"同题跨轮差分门槛{'通过' if rr['same_gate'] else '未通过'}。跨数字供体门槛{'通过' if rr['disjoint_gate'] else '未通过'}。"])
    r=load(ROOT/"SPECIFICITY_SUMMARY.json")
    eq=r["repair"]["outcomes"]
    safe=r["safe"]["outcomes"]
    d.extend(["","## 相同剂量作用于同答案安全受体","",
              f"两类同答案受体都使用来自借位题的同一个原生跨轮向量，剂量相同。K2 精确多 100 在借位受体为 {eq['same_vector']['K2_foil']}/{r['repair']['n']}，在无需借位受体为 {safe['same_vector']['K2_foil']}/{r['safe']['n']}；匹配几何对应 {eq['geometry']['K2_foil']} 与 {safe['geometry']['K2_foil']}。安全组新增 K2 错误 {safe['same_vector']['K2_harm']} 道。双重配对差为 {paired(r['borrow_excess_K2_foil_vs_safe'])}（97.5%区间）。"])
    x=load(ROOT/"PATH_SUMMARY.json")
    d.extend(["","## 第14层 A 路径和第17层交互的原生因果检验","",
              "| K2 条件 | 正解 | 同题原 +100 | 同答案安全组新增错误 |","|---|---:|---:|---:|"])
    for i,v in x["repair"]["outcomes"].items():
        d.append(f"| {i} | {v['correct']}/{n} | {v['foil']}/{n} | {x['safe']['outcomes'][i]['wrong_new']}/{x['safe']['n']} |")
    profile=load(ROOT/"TRAJECTORY_SUMMARY.json")
    factors=[profile[label]["statistics"]["K2"][1] for label in ("repair","safe")]
    d.extend(["",f"在 A 读入回拨后恰好重现原错答的 {x['exact_reversions']} 道中，恢复各分量的完整正解数为 {x['rescues']}。交互分量恢复的配对差为 {paired(x['interaction_rescue_excess'])}（97.5%区间）。完整读出恢复是结构校验。路径链门槛{'通过' if x['chain_gate'] else '未通过'}。",
              "","## 物理层输入上的原生交互几何","","![原生四角读出交互范数](results/TRAJECTORY.png)","",
              f"曲线为中位数，阴影为四分位范围。第 14 层至第 17 层，K2 交互范数中位数在修复组由 {factors[0][13]:.2f} 增至 {factors[0][16]:.2f}，安全组由 {factors[1][13]:.2f} 增至 {factors[1][16]:.2f}。两类数值编辑的总幅度不同，原始交互范数用于描述层间变化。所有完整解码、原生匹配、控制剂量、锁定协议与哈希见 results/。",""])
    Path("REPORT_ZH.md").write_text("\n".join(d),encoding="utf-8")


if __name__=="__main__":
    print(verify(),flush=True)
    render()

