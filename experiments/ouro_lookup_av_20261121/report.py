"""Verify the independent lookup cohort, factorization audit, and paired statistics."""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import transformers

from av_deck import DOMAINS, build, repair, summary
from av_stats import ARMS, DIRECTIONS, PRIMARY, summarize
from av_test import auditors
from native import MODEL_HASH, REV, load, save, sha
from rule import MAX_NEW, parse

R = Path("results")
LABEL = {"route":"仅路由 A", "value":"仅内容 V", "joint":"联合 A/V", "minus_interaction":"联合减交互",
         "minus_control":"联合减匹配对照", "identity":"受体自身"}
CONTRAST = {"joint_route":"联合 − 仅路由", "joint_value":"联合 − 仅内容", "interaction":"减匹配对照 − 减交互",
            "route_identity":"仅路由 − 受体自身", "value_identity":"仅内容 − 受体自身", "control_cost":"联合 − 减匹配对照"}


def equal(actual,expected):
    assert json.loads(json.dumps(actual))==expected


def answer(x):
    assert 0<len(x["tokens"])<=MAX_NEW and x["number"]==parse(x["text"])


def verify_native():
    d,n,z = [load(R/name) for name in ("DECK.json","NATIVE.json","NATIVE_SUMMARY.json")]
    equal(build(),d)
    assert n["deck_sha256"]==sha(R/"DECK.json")
    rows = n["rows"]
    for x,y in zip(d["rows"],rows,strict=True):
        assert x=={k:v for k,v in y.items() if k!="native"}
        assert len(y["native"]["original"])==2
        for a in y["native"]["original"]:
            answer(a)
    equal(summary(rows),z)
    v = dict(status="PASS",model_revision=REV,weights_sha256=os.environ["OURO_ACTUAL_SHA256"],
             expected_weights_sha256=MODEL_HASH,gpu=os.environ["OURO_GPU"],torch=torch.__version__,
             transformers=transformers.__version__,cuda=torch.version.cuda,deck_sha256=sha(R/"DECK.json"),
             native_sha256=sha(R/"NATIVE.json"),prompts=len(rows))
    assert v["weights_sha256"]==MODEL_HASH
    return rows,z,v


def verify():
    rows,z,v = verify_native()
    if not z["gate"]:
        return v
    cohort,raw = [load(R/name) for name in ("COHORT.json","AV_RAW.json")]
    assert cohort["native_sha256"]==v["native_sha256"] and raw["cohort_sha256"]==sha(R/"COHORT.json")
    selected = [r for r in rows if repair(r)]
    expected = auditors(selected)
    assert cohort["audit"]==expected and cohort["repair"]==[r["id"] for r in selected]==[r["id"] for r in raw["rows"]]
    audited = set().union(*expected.values())
    for x,y in zip(selected,raw["rows"],strict=True):
        assert x=={k:y[k] for k in x} and y["old"]==x["native"]["original"][0]["number"]
        assert set(y["pred"])==set(DIRECTIONS)
        for d,n in zip(DIRECTIONS,x["native"]["original"],strict=True):
            assert set(y["pred"][d])==set(ARMS) and y["pred"][d]["identity"]==n
            for a in y["pred"][d].values():
                answer(a)
        assert ("audit" in y)==(y["id"] in audited)
        if "audit" in y:
            a = y["audit"]
            assert a["checks"]==72 and all(0<=q<.001 for k,q in a.items() if k!="checks")
    equal(summarize(raw["rows"]),load(R/"AV_SUMMARY.json"))
    v.update(raw_sha256=sha(R/"AV_RAW.json"),cohort_sha256=raw["cohort_sha256"],runtime_s=raw["elapsed"],
             peak_allocated_mib=raw["peak_allocated_mib"])
    return v


def plot(z):
    fig,ax = plt.subplots(figsize=(9.4,4.1))
    labels = ["Route only","Value only","Joint","Joint − interaction","Joint − control","Identity"]
    for i,d in enumerate(DOMAINS):
        data = z["domains"][d]
        ax.bar([j+.32*(i-.5) for j in range(len(ARMS))],[100*data["arms"][k]["dual"]/data["n"] for k in ARMS],
               .31,label=d.title(),color=("#3A688A","#A55243")[i])
    ax.set_xticks(range(len(labels)),labels)
    ax.set_ylabel("Exact bidirectional answer transfer (%)")
    ax.legend(frameon=False)
    ax.spines[["right","top"]].set_visible(False)
    ax.grid(axis="y",alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    dst = R/"ROUTE_CONTENT.png"
    fig.savefig(dst,dpi=180,bbox_inches="tight")
    plt.close(fig)
    with Image.open(dst) as i:
        assert i.width>1000 and i.height>450
        i.verify()
    return sha(dst)


def render(v):
    n = load(R/"NATIVE_SUMMARY.json")
    d = ["# 查表精确纠正中的注意力路由、内容与乘积", "",
         "冻结 Ouro-1.4B Base。在同一个生成前缀上捕获两轮原生注意力权重 A 和内容 V，交换第 12–17 层所有有效位置的对应因子。直接在投影后的注意力向量上移除交互或匹配向量，随后模型原生归一化、MLP 和后缀自由重算。用同一题在 K1 生成真值、K2 重现原错表值作双向整答指标。", "",
         "| 格式 | 题数 | 错→错 | 错→对 | 对→错 | 对→对 | 符合精确纠正条件 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for format in DOMAINS:
        x=n[format]
        cells=" | ".join(str(x["transitions"][k]) for k in ("00","01","10","11"))
        d.append(f"| {format} | {x['n']} | {cells} | {x['repairs']} |")
    if not n["gate"]:
        d.extend(["", "两种格式的自然队列未同时达到预设 48 题门槛。"])
    else:
        z=load(R/"AV_SUMMARY.json")
        d.extend(["", "| 格式 | 条件 | n | K1 正答 | K2 原错复现 | K2 保持正答 | 双向整答 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
        for format in DOMAINS:
            x=z["domains"][format]
            for k,y in x["arms"].items():
                d.append(f"| {format} | {LABEL[k]} | {x['n']} | {y['rescue']} | {y['damage']} | {y['k2_correct']} | {y['dual']} |")
        d.extend(["", "| 格式 | 配对比较 | 多命中 / 少命中 | 效应及区间（百分点） | 用途 |",
                  "|---|---|---:|---:|---|"])
        for format in DOMAINS:
            for k,p in z["domains"][format]["contrasts"].items():
                a,b=[100*t for t in p["interval"]]
                d.append(f"| {format} | {CONTRAST[k]} | {p['extra']} / {p['loss']} | {100*p['mean']:+.1f} [{a:+.1f}, {b:+.1f}] | {'主比较，同时区间' if k in PRIMARY else '描述，逐项区间'} |")
        upper = [100*z["domains"][k]["contrasts"]["interaction"]["upper_simultaneous"] for k in DOMAINS]
        decision = "通过" if z["supported"] else "未通过"
        stop = "达到" if z["stop"] else "未达到"
        d.extend(["", f"本次干预的交互贡献判据{decision}，预设停止条件{stop}。标准和新格式中交互额外命中的校正单侧上界为 {upper[0]:.2f} 和 {upper[1]:.2f} 个百分点。", "",
                  "消融针对跨轮交换中定义的乘积项。完整路由或内容可携带已形成的答案状态。要确定原生 K2 是否依赖某个分量，需要在未实施联合交换的 K2 上单独验证。", "",
                  f"固定 {z['audit']['n']} 道题核验了 {z['audit']['checks']} 个注入点。原生投影重建最大误差 {z['audit']['native_projection']:.3g}，因子恒等式最大误差 {z['audit']['factorization']:.3g}，独立逐层后缀重算最大误差 {z['audit']['suffix']:.3g}，自身回放末端最大误差 {z['audit']['identity']:.3g}。所有自身回放均恢复原生自由生成。", "",
                  "![注意力因子的双向整答](results/ROUTE_CONTENT.png)","",
                  "先前同题回放已确认第 12–17 层注意力写入的双向作用。更早的四头第 6 层实验用完整候选平均对数似然评估固定路由，其规则乘积效应未通过。本轮覆盖六层所有头，并以不受限生成的完整整数计分。两条实验链分别见[中层写入](../ouro_lookup_midwrites_20261121/REPORT_ZH.md)与[固定路由与受体内容](../ouro_route_content_20261121/REPORT_ZH.md)。"])
        v["figure_sha256"]=plot(z)
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__ == "__main__":
    v = verify()
    render(v)
    save(R/"VERIFICATION.json",v)
    print(v,flush=True)
