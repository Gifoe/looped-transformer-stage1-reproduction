"""Verify frozen input and output invariants, then render measured results."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import platform

import torch
import transformers

from bidi import baseline, event, prior_pairs, unordered
from carry import PROMPT, numeric
from native import MODEL_HASH, REV, load, save, sha
from operand import MODES, count_results


def check_text(out):
    assert out["tokens"] and numeric(out["text"])==out["number"]


def verification(root,p_path,b_path):
    m=load(root/"DECK.json")
    p=load(root/"NATIVE.json")
    assert m["sources"]==[sha(p_path),sha(b_path)]
    assert m["prompt"]==PROMPT and p["deck_sha256"]==sha(root/"DECK.json")
    assert len(m["rows"])==len(p["rows"]) and len(m["midpoint_splits"])==300
    seen=prior_pairs(p_path,b_path)
    prior_size=len(seen)
    arm_count=Counter()
    occupied=defaultdict(lambda: [set(),set()])
    n_by_split=Counter()
    other_common=Counter()
    parse_failure=Counter()
    for a,b in zip(m["rows"],p["rows"],strict=True):
        assert a=={k:x for k,x in b.items() if k!="pred"}
        x,y=a["pairs"]
        i=a["changed_operand"]
        assert y[i]-x[i]==200 and x[i]%100==y[i]%100 and x[1-i]==y[1-i]
        assert list(map(sum,a["pairs"]))==a["target"]==[a["midpoint"]-100,a["midpoint"]+100]
        assert a["split"]==m["midpoint_splits"][str(a["midpoint"])]
        for o in (x,y):
            assert unordered(o) not in seen and all(100<=v<=999 for v in o)
            seen.add(unordered(o))
        assert len(b["pred"])==2 and all(len(z)==2 for z in b["pred"])
        for round_ in b["pred"]:
            for generation in round_:
                check_text(generation)
        n_by_split[a["split"]]+=1
        guesses=[x["number"] for x in b["pred"][0]]
        if guesses[0] is not None and guesses[0]==guesses[1] and guesses[0] not in a["target"] and guesses[0]!=a["midpoint"]:
            other_common[a["split"]]+=1
        for k,gen in enumerate(b["pred"],1):
            parse_failure[f"{a['split']}_K{k}"]+=sum(x["number"] is None for x in gen)
        arm_count[a["split"],i]+=1
        for j,pred in enumerate(b["pred"][0]):
            if pred["number"]==a["midpoint"]:
                occupied[a["split"],a["midpoint"]][j].add(a["id"])
    assert len(seen)-prior_size==len(p["rows"])*2
    measured=load(root/"NATIVE_SUMMARY.json")
    assert json.loads(json.dumps(baseline(p["rows"],root)))==measured
    n_ticks={}
    one_sign={}
    for s in n_by_split:
        bins=[v for (sp,m),v in occupied.items() if sp==s]
        n_ticks[s]=sum(bool(v[0] and v[1]) for v in bins)
        one_sign[s]=[sum(bool(v[i]) for v in bins) for i in (0,1)]
    detail=dict(status="PASS", model_revision=REV, model_sha256=MODEL_HASH, python=platform.python_version(),
                torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
                gpu="NVIDIA GeForce RTX 4090", weights_as_run="BF16 checkpoint promoted to FP32; no TF32 or autocast",
                deck_sha256=sha(root/"DECK.json"), raw_native_sha256=sha(root/"NATIVE.json"),
                unique_unordered_problems=len(seen)-prior_size,stems_by_split=n_by_split,
                common_midpoint_even_across_stems=n_ticks, midpoint_bins_per_sign=one_sign,
                other_common_wrong_reply=other_common, parsed_failures=parse_failure)
    source=root/"OPERAND_RAW.json"
    operand=load(source)
    assert operand["native_sha256"]==sha(root/"NATIVE.json")
    lookup={j["id"]:j for j in p["rows"]}
    assert len(operand["rows"])==300 and len({x["midpoint"] for x in operand["rows"]})==300
    eps=[]
    for row in operand["rows"]:
        native=lookup[row["id"]]
        assert row["midpoint"]==native["midpoint"] and row["split"]==native["split"] and native["changed_operand"]==0
        if not row["aligned"]:
            continue
        eps.append(row["max_geometry_abs_error"])
        assert len(row["cases"])==2 and set(row["modes"])==set(MODES)
        for k,z in enumerate(row["cases"]):
            assert z["recipient"]==native["pairs"][k] and z["native"]==[r[k] for r in native["pred"]]
            assert sum(z["recipient"])==z["answer"]
            assert z["donor"][0]==native["pairs"][1-k][0]
            assert abs(z["donor"][1]-z["recipient"][1])==100
            assert z["counterfactual"]==z["donor"][0]+z["recipient"][1] and sum(z["donor"])==z["donor_answer"]
            assert len({z["answer"],z["counterfactual"],z["donor_answer"]})==3
            assert z["digits"] and set(z["digits"]).issubset(z["prefix"])
            for v in row["modes"].values():
                check_text(v[k])
            assert row["modes"]["native"][k]["tokens"]==z["native"][1]["tokens"]
    assert load(root/"OPERAND_SUMMARY.json")==count_results(m["midpoint_splits"],operand["rows"])
    detail.update(operand_raw_sha256=sha(source),maximum_geometry_error=max(eps,default=0))
    if (root/"CHANNEL_RAW.json").exists():
        from channels import summarize
        assay=load(root/"CHANNEL_RAW.json")
        assert assay["operand_sha256"]==sha(source)
        o_src={x["id"]:x for x in operand["rows"]}
        for r in assay["rows"]:
            b=o_src[r["id"]]
            assert b["aligned"] and r["case"]==b["cases"]
            for values in r["outputs"].values():
                for v in values:
                    check_text(v)
            for key in ("native","prefix","both"):
                want=b["modes"]["native" if key=="both" else key]
                assert [q["tokens"] for q in r["outputs"][key]]==[x["tokens"] for x in want]
        expect=load(root/"CHANNEL_SUMMARY.json")
        assert summarize(assay["rows"],root)==expect
        detail["channel_raw_sha256"]=sha(root/"CHANNEL_RAW.json")
    if (root/"STATE_SUMMARY.json").exists():
        t=load(root/"STATE_TRANSFER.json")
        assert t["native_sha256"]==sha(root/"NATIVE.json")
        assert all(event(lookup[j["id"]])["direct"] for j in t["rows"])
        detail["state_sha256"]=sha(root/"STATE_TRANSFER.json")
    save(root/"VERIFICATION.json",detail)
    return detail


def pct(x):
    return f"{100*x:+.2f}"


def render(root,detail):
    b=load(root/"NATIVE_SUMMARY.json")["confirmation"]
    e=load(root/"OPERAND_SUMMARY.json")["confirmation"]
    chp=root/"CHANNEL_SUMMARY.json"
    s=b
    m=e["modes"]
    doc=["# Ouro-1.4B Base 跨循环保留与重用的操作数信息","",
         "同一冻结模型、同一组六个加法示例、同一完整词表贪心解码。评估首个非空行完整整数。题组以中间整数隔离发现和确认。所有运算数无序对与先前两组加法试验及示例互斥。","",
         "## 同一 K1 错答能否向相反方向修复","",
         f"确认集 {b['n_midpoints']} 个中间整数、{s['drawn_stems']} 个严格配对。对内保持一个完整操作数及另一个操作数的低两位。原生 K1、K2 分别答对 {s['k1_correct']}、{s['k2_correct']} 道（每个深度 {2*s['drawn_stems']} 道）。",
         f"两道 K1 都生成预定中间错答的有 **{s['eligible']}** 个严格配对，覆盖 {b['distinct_eligible_midpoints']} 个中间整数。K2 在 {s['direct']} 对按各自正确方向完整答对，交换 K2 输出后为 {s['swap']} 对。严格门槛{'通过' if b['pass_gate'] else '未通过'}。",
         f"低真值一侧、高真值一侧的入口计数分别为 {s['low_k1_mid']}、{s['high_k1_mid']} 次（允许重合）。允许跨题干按同一错答重配时，双方分别出现的中间整数共有 {detail['common_midpoint_even_across_stems']['confirmation']} 个。另有 {detail['other_common_wrong_reply'].get('confirmation',0)} 对共享其他错误整数。跨题干与其他错答结果为描述性诊断。",
         "",
         "## 保留 K1 读出，换入另一个第一操作数","",
         "原题 A+B 的第二轮输入只替换第一操作数到第二操作数之前的连续 H1 状态，供体是 A*+B*。受体的末提示位置及生成位置仍使用自身 H1。B 与 B* 相差 100，并核验在供体前缀的 H1 相同。植入内容未读到 B*。实验区分受体原答案 A+B、目标 A*+B 与供体完整答案 A*+B*。","",
         f"确认集 {e['n_bins']} 个题组，{e['aligned_bins']} 个满足 token 对齐。主比较为两个方向都实际生成完整反事实整数，随机对照逐位置匹配替换向量范数与对原向量角度。","",
         "| 处理 | 生成 A*+B | 同组双向生成 A*+B | 生成 A*+B* | 生成 A+B | 原生 K1→K2 纠错题中生成 A*+B |",
         "|---|---:|---:|---:|---:|---:|"]
    names={"native":"原生复跑","prefix":"整段自然状态","prefix_noise":"整段等几何对照","digits":"仅变化数字位置","digits_noise":"数字位置等几何对照"}
    for arm in MODES:
        x=m[arm]
        keys=("counterfactual","two_counterfactual","donor_answer","answer")
        doc.append("| "+names[arm]+" | "+" | ".join(str(x[k]) for k in keys)+
                   f" | {x['counterfactual_among_native_repairs']}/{x['native_repairs']} |")
    c=m["prefix"]["two_switch_excess"]
    doc.extend(["",f"双向生成的配对超额为 {pct(c['mean'])} 个百分点，95% 区间 [{pct(c['interval'][0])}, {pct(c['interval'][1])}]，{'通过' if e['pass_primary'] else '未通过'}独立预设门槛。逐题原始 token 序列与控制范数检查见 results/。"])
    descript=load(root/"OPERAND_DESCRIPTIVE.json")
    assert descript["operand_sha256"]==sha(root/"OPERAND_RAW.json")
    d=descript["by_split"]["confirmation"]
    low_carry=sum(v for k,v in d.items() if k.endswith("carry=True/n"))
    exact_by_sign={z:sum(v for k,v in d.items() if k.startswith(z+"/") and k.endswith("/prefix")) for z in ("+200","-200")}
    doc.extend(["",f"按预先固定的首个题干取样后，确认集有 {low_carry}/{2*e['aligned_bins']} 个方向涉及向百位进位。移植后的 +200、−200 正确切换分别为 {exact_by_sign['+200']}、{exact_by_sign['-200']} 次。本轮支持加数状态参与第二轮作答，结论限定在所测数值差与进位覆盖。原生 K2 修复子集的移植结果属于描述性分析。"])
    if chp.exists():
        ch=load(chp)["confirmation"]
        if ch["n"]:
            doc.extend(["","## K2 注意力通路替换","",
                       f"在已显示两向自然前缀移植都生成反事实的题中，预设抽样 {ch['n']} 个确认组。在第二轮每层，把被换入位置的 K、V 分别或同时恢复成原生受体值，做完整解码。","",
                       "| 第二轮通路 | 双向恢复受体正确整数 | 单方向数（受体/反事实/供体/其他） |",
                       "|---|---:|---:|"])
            for a,t in ch["treatments"].items():
                doc.append(f"| {a} | {t['both_recipient']} | {t['recipient']} / {t['counterfactual']} / {t['donor']} / {t['other']} |")
            for a in ("keys","values"):
                z=ch["treatments"][a]["both_restoration_excess"]
                doc.extend(["",f"{a} 对匹配控制的双向原答案恢复率配对差为 {pct(z['mean'])} 个百分点，97.5% 区间 [{pct(z['ci'][0])}, {pct(z['ci'][1])}]。"])
            doc.extend(["","K+V 联合恢复检验注意力的结构闭合。单分量估计属于全层路径干预。分辨具体层、注意力头、后续 MLP 或加法算法仍需独立实验。"])
    if (root/"STATE_SUMMARY.json").exists():
        doc.extend(["","共同错答组的独立状态移植结果另见 results/STATE_SUMMARY.json。"])
    doc.extend(["","## 复现","","各协议及源码哈希见 results/。数据验证重新计算所有主结论和分母，核对无序题对排重及原生贪心序列。",""])
    Path("REPORT_ZH.md").write_text("\n".join(doc),encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("results"))
    p.add_argument("--pilot",type=Path,default=Path("/root/ouro-open-carry-20261121/MANIFEST.json"))
    p.add_argument("--boundary",type=Path,default=Path("/root/ouro-carry-boundary-20261121/results/MANIFEST.json"))
    a=p.parse_args()
    render(a.root,verification(a.root,a.pilot,a.boundary))

