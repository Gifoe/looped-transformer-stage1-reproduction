"""Verify table worlds, complete answers and the paired conflict analysis."""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import transformers

from conf_deck import DOMAINS,build,summary as native_summary
from conf_pair import SIDES,build as pairs_build,qualified,spans,summary as pair_summary
from conf_stats import ARMS,DIRECTIONS,PATHS,summarize
from conf_test import auditors
from native import MODEL_HASH,REV,load,save,sha
from rule import MAX_NEW,parse

R = Path("results")
LABEL = {"value":"目标行 V","value_geometry":"V 几何","value_unrelated":"V 无关行",
         "carry":"入轮残差","carry_geometry":"残差几何","carry_unrelated":"残差无关行",
         "both":"共同交换","clean":"自身回放"}


def score(a):
    assert 0<len(a["tokens"])<=MAX_NEW and parse(a["text"])==a["number"]


def verify():
    d,n,z,p,e = [load(R/x) for x in ("DECK.json","NATIVE.json","NATIVE_SUMMARY.json","PAIRS.json","PAIR_SUMMARY.json")]
    assert d==build() and n["deck_sha256"]==sha(R/"DECK.json") and p["native_sha256"]==sha(R/"NATIVE.json")
    for a,b in zip(d["rows"],n["rows"],strict=True):
        assert a=={k:v for k,v in b.items() if k!="native"}
        assert len(b["native"]["original"])==2
        for a0 in b["native"]["original"]:
            score(a0)
    assert z==native_summary(n["rows"])
    tok = transformers.AutoTokenizer.from_pretrained(Path(os.environ["OURO_MODEL_PATH"]),trust_remote_code=True,local_files_only=True)
    for x,y in zip(pairs_build(n["rows"]),p["rows"],strict=True):
        assert x=={k:y[k] for k in x}
        assert set(y["pair_native"])==set(SIDES) and y["pair_native"]["original"]==x["native"]["original"][1]
        assert spans(tok,y)==y["span"]
        for a in y["pair_native"].values():
            score(a)
    assert e==pair_summary(p["rows"])
    v = dict(status="PASS",model_revision=REV,expected_weights_sha256=MODEL_HASH,
             weights_sha256=os.environ["OURO_ACTUAL_SHA256"],gpu=os.environ["OURO_GPU"],torch=torch.__version__,
             transformers=transformers.__version__,cuda=torch.version.cuda,prompts=len(d["rows"]),
             deck_sha256=sha(R/"DECK.json"),native_sha256=sha(R/"NATIVE.json"),pair_sha256=sha(R/"PAIRS.json"))
    assert v["weights_sha256"]==MODEL_HASH
    if not e["gate"]:
        return v
    c,r = [load(R/x) for x in ("COHORT.json","CONFLICT_RAW.json")]
    assert c["pair_sha256"]==v["pair_sha256"] and r["cohort_sha256"]==sha(R/"COHORT.json")
    enroll = [x for x in p["rows"] if qualified(x)]
    assert c["cases"]==[x["id"] for x in enroll]==[x["id"] for x in r["rows"]]
    assert c["audit"]==auditors(enroll)
    audit = {i for d in c["audit"].values() for g in d.values() for i in g}
    for x,y in zip(enroll,r["rows"],strict=True):
        assert {k:v for k,v in x.items() if k!="span"}=={k:y[k] for k in x if k!="span"}
        assert set(y["pred"])==set(DIRECTIONS)
        for s in DIRECTIONS:
            o = y["pred"][s]
            assert set(o)==set(ARMS) and o["clean"]==x["pair_native"][s]
            for a in o.values():
                score(a)
        assert ("audit" in y)==(y["id"] in audit)
        if "audit" in y:
            a=y["audit"]
            assert (a["v_sites"],a["carry_sites"],a["tokens"])==(240,10,16)
            assert 0<=a["cache_causal"]<.001 and 0<=a["attention"]<.001
    assert json.loads(json.dumps(summarize(r["rows"])))==load(R/"CONFLICT_SUMMARY.json")
    assert set(r["fallback"])==set(PATHS) and all(0<=x<=n and n>0 for x,n in r["fallback"].values())
    v.update(raw_sha256=sha(R/"CONFLICT_RAW.json"),cohort_sha256=r["cohort_sha256"],
             peak_allocated_mib=r["peak_allocated_mib"],runtime_s=r["elapsed"],fallback=r["fallback"])
    return v


def plot(z):
    fig,ax = plt.subplots(figsize=(9.8,4.1))
    keys = ("value","value_geometry","value_unrelated","carry","carry_geometry","carry_unrelated","both")
    names = ("Row V","V geometry","V unrelated","Carry","Carry geometry","Carry unrelated","Joint")
    for i,d in enumerate(DOMAINS):
        q = z["domains"][d]["repair"]
        ax.bar([j+.32*(i-.5) for j in range(len(keys))],[100*q["counts"][a]["dual"]/q["n"] for a in keys],
               .31,label=d.title(),color=("#3A688A","#A55243")[i])
    ax.set_xticks(range(len(names)),names)
    ax.set_ylabel("Exact bidirectional transfer (%)")
    ax.legend(frameon=False)
    ax.spines[["right","top"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y",alpha=.2)
    fig.tight_layout()
    p=R/"TABLE_CONFLICT.png"
    fig.savefig(p,dpi=180,bbox_inches="tight")
    plt.close(fig)
    with Image.open(p) as i:
        assert i.width>1000 and i.height>450
        i.verify()
    return sha(p)


def effect(x):
    if x is None:
        return "无样本"
    lo,hi = (100*y for y in x["interval"])
    return f"{100*x['mean']:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def render(v):
    n,p = [load(R/x) for x in ("NATIVE_SUMMARY.json","PAIR_SUMMARY.json")]
    d=["# 目标表项 V 与入轮答前残差的冲突", "",
       "冻结 Ouro-1.4B Base。两张表只交换目标值与一个合法替代值，保留查询键。在当前生成前缀，从另一表移入 K2 各层目标行数字位置的 V，或移入 K2 起点的最后位置残差。单项交换让两个来源分别指向不同值。K2 的 Q、K、注意力权重和其余计算随干预状态重算。主终点要求同一题两个方向的自由生成均输出完整供体表值。", "",
       "| 格式 | 原始题数 | 精确自然纠正 | 纠正对齐 / 合格 | 稳定题样本 | 稳定对齐 / 合格 |",
       "|---|---:|---:|---:|---:|---:|"]
    for k in DOMAINS:
        q=p[k]
        d.append(f"| {k} | {n[k]['n']} | {n[k]['repairs']} | {q['repair']['aligned']} / {q['repair']['eligible']} | {q['stable']['n']} | {q['stable']['aligned']} / {q['stable']['eligible']} |")
    if not p["gate"]:
        d.extend(["", "至少一个格式的合格纠正题对少于预设 48 道，未启动因果阶段。"])
    else:
        z = load(R/"CONFLICT_SUMMARY.json")
        d.extend(["", "| 格式 | 题组 | 干预 | 原表供体 / 受体 / 其他 | 交换表供体 / 受体 / 其他 | 双向供体 |", "|---|---|---|---:|---:|---:|"])
        for fmt in DOMAINS:
            for kind in ("repair","stable"):
                for k,y in z["domains"][fmt][kind]["counts"].items():
                    x=[" / ".join(str(y[s][u]) for u in ("donor","recipient","other")) for s in DIRECTIONS]
                    d.append(f"| {fmt} | {'纠正' if kind=='repair' else '稳定'} | {LABEL[k]} | {x[0]} | {x[1]} | {y['dual']} |")
        d.extend(["", "| 格式 | 路径 | 对照 | 多 / 少双向命中 | 同时区间（百分点） | 校正上界（百分点） |", "|---|---|---|---:|---:|---:|"])
        for fmt in DOMAINS:
            for k,comparisons in z["domains"][fmt]["repair"]["tests"].items():
                for c,x in comparisons.items():
                    d.append(f"| {fmt} | {LABEL[k]} | {LABEL[k+'_'+c]} | {x['extra']} / {x['loss']} | {effect(x)} | {100*x['upper']:.2f} |")
        supported="、".join(LABEL[k] for k in z["supported"]) or "无"
        excluded="、".join(LABEL[k] for k in z["excluded"]) or "无"
        d.extend(["", f"两种格式均通过定向双向转移门槛的路径为 {supported}。两种格式均排除相对对照达 10 个百分点的路径为 {excluded}。"])
        for fmt in DOMAINS:
            q=z["domains"][fmt]["repair"]
            d.append(f"{fmt} 中目标行 V 减入轮残差的双向效果差为 {effect(q['value_vs_carry'])} 个百分点，使用描述性逐项区间。")
        audit=z["audit"]
        f=v["fallback"]
        d.extend(["", f"无关行位移为零时回退到几何控制。V 有 {f['value'][0]:,}/{f['value'][1]:,} 个头与位置回退，残差有 {f['carry'][0]:,}/{f['carry'][1]:,} 个位置回退。",
                  f"固定 {audit['n']} 个题对核验 {audit['v_sites']} 次 V 安装、{audit['carry_sites']} 次残差安装和 {audit['tokens']} 次生成位置上的实际 AV。仅改最后位置残差时，较早目标行 V 的最大偏差为 {audit['cache_causal']:.3g}，实际 AV 独立核算最大偏差 {audit['attention']:.3g}。所有自身回放均复现干净整答。", "",
                  "按实际生成前缀反复移植可检验携带内容的可用性。几何匹配在每条路径内部完成。目标行操作覆盖多层多位置，入轮残差操作只覆盖一个位置，因此两路径间的整答率差是本实验条件下的比较。原生必要性和贡献比例需另证。", "",
                  "![目标行值和入轮残差的竞争](results/TABLE_CONFLICT.png)", "",
                  "先前跨表残差实验在固定后续写入时，第 18 层入口首次同时通过跨轮和跨表标准，新格式在入轮已有单独的跨表信号。本轮后续写入随干预状态重算，并将入轮最后位置与目标行数字位置分开。相关协议与原始数据见[五个层界的残差交换](../ouro_lookup_onset_20261121/REPORT_ZH.md)。"])
        v["figure_sha256"]=plot(z)
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__ == "__main__":
    v=verify()
    render(v)
    save(R/"VERIFICATION.json",v)
    print(v,flush=True)
