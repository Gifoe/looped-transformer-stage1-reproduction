"""Verify enrollment, exact endpoints and branch-replay evidence, then publish a report."""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import transformers

from edge import short_hash
from mid_deck import DOMAINS, build, control, history, repair, summary
from mid_stats import ARMS, DIRECTIONS, MODES, summarize
from native import MODEL_HASH, REV, load, save, sha
from rule import MAX_NEW, parse

R = Path("results")
NAMES = {"attention":"注意力", "mlp":"MLP", "both":"联合"}


def exact(value):
    assert 0<len(value["tokens"])<=MAX_NEW and parse(value["text"]) == value["number"]


def equal(a,b):
    assert json.loads(json.dumps(a)) == b


def verify():
    deck = load(R/"DECK.json")
    native = load(R/"NATIVE.json")
    equal(build(), deck)
    assert native["deck_sha256"] == sha(R/"DECK.json")
    original = native["rows"]
    for x,y in zip(deck["rows"],original,strict=True):
        assert x == {k:v for k,v in y.items() if k!="native"}
        assert len(y["native"]["original"]) == 2
        for z in y["native"]["original"]:
            exact(z)
    n0 = load(R/"NATIVE_SUMMARY.json")
    equal(summary(original), n0)
    v = dict(status="PASS", model_revision=REV, expected_weights_sha256=MODEL_HASH,
             weights_sha256=os.environ["OURO_ACTUAL_SHA256"], gpu=os.environ["OURO_GPU"],
             torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
             deck_sha256=sha(R/"DECK.json"), native_sha256=sha(R/"NATIVE.json"), prompts=len(original))
    assert v["weights_sha256"] == MODEL_HASH
    if not n0["gate"]:
        return v
    cohort,raw = [load(R/f) for f in ("COHORT.json","MID_RAW.json")]
    assert cohort["native_sha256"] == v["native_sha256"] and raw["cohort_sha256"] == sha(R/"COHORT.json")
    assert 0<=raw["fallback"]<raw["positions"] and len(raw["rows"])==len(cohort["repair"])
    from mid_deck import prompt
    seen = history() | {prompt(q,"original") for q in original}
    eligible = [control(x) for x in original if repair(x)]
    assert cohort["repair"] == [r["id"] for r in raw["rows"]] == [r["id"] for r in eligible]
    expected = {d:sorted((r["id"] for r in eligible if r["domain"]==d),key=lambda x:short_hash("midwrites audit",x))[:8] for d in DOMAINS}
    assert cohort["audit"] == expected
    audit_ids = set().union(*expected.values())
    for r,q in zip(raw["rows"],eligible,strict=True):
        assert {k:r[k] for k in q} == q
        alt = prompt(q,"unrelated_original")
        assert alt not in seen
        seen.add(alt)
        assert r["identity"] == dict(zip(DIRECTIONS,q["native"]["original"],strict=True))
        assert set(r["pred"]) == set(MODES)
        for directions in r["pred"].values():
            assert set(directions) == set(DIRECTIONS)
            for arms in directions.values():
                assert set(arms) == set(ARMS)
                for value in arms.values():
                    exact(value)
        assert ("audit" in r) == (r["id"] in audit_ids)
        if "audit" in r:
            assert r["audit"]["checks"] == 72 and 0<=r["audit"]["suffix"]<.001 and 0<=r["audit"]["identity"]<.001
    z = load(R/"MID_SUMMARY.json")
    equal(summarize(raw["rows"]), z)
    v.update(raw_sha256=sha(R/"MID_RAW.json"), cohort_sha256=raw["cohort_sha256"],
             runtime_s=raw["elapsed"],peak_allocated_mib=raw["peak_allocated_mib"],
             unrelated_geometry_fallback=raw["fallback"],unrelated_token_writes=raw["positions"])
    return v


def plot(z):
    fig, panels = plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    colors = ("#3A688A","#A55243","#7D8183")
    for panel,d in zip(panels,DOMAINS,strict=True):
        for i,a in enumerate(ARMS):
            values = [z["domains"][d][m] for m in MODES]
            panel.bar([j+.23*(i-1) for j in range(3)],
                      [100*r["counts"][a]["dual"]/r["n"] for r in values],.22,label=a,color=colors[i])
        panel.set_xticks(range(3),["Attention","MLP","Both"])
        panel.set_title(d.title(),loc="left")
        panel.legend(frameon=False,fontsize=8)
        panel.spines[["right","top"]].set_visible(False)
        panel.grid(axis="y",alpha=.2)
        panel.set_axisbelow(True)
    panels[0].set_ylabel("Exact bidirectional answer transfer (%)")
    fig.tight_layout()
    dst = R/"MIDDLE_WRITES.png"
    fig.savefig(dst,dpi=180,bbox_inches="tight")
    plt.close(fig)
    with Image.open(dst) as p:
        assert p.width>1000 and p.height>450
        p.verify()
    return sha(dst)


def render(v):
    n = load(R/"NATIVE_SUMMARY.json")
    d = ["# 第 12–17 层原生写入能否双向改变查表答案", "",
         "在冻结的 Ouro-1.4B Base 上，两轮都从受体本来的状态运行。只交换第 12–17 层有效前缀中的指定注意力或 MLP 写入，第 18–24 层自由重算。每次生成按各干预实际产生的前缀重新采集供体写入。双向命中要求同一道题的 K1 生成真值，K2 重现原来的错误表值。", "",
         "| 格式 | 总题数 | 错→错 | 错→对 | 对→错 | 对→对 | 合格自然纠正 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for k in DOMAINS:
        a = n[k]
        cells = " | ".join(str(a["transitions"][q]) for q in ("00","01","10","11"))
        d.append(f"| {k} | {a['n']} | {cells} | {a['repairs']} |")
    if not n["gate"]:
        d.extend(["", "原生自然纠正样本数未达到两个格式的预设门槛。"])
    else:
        z = load(R/"MID_SUMMARY.json")
        d.extend(["", "| 格式 | 写入 | n | 真实恢复 / 回退 / 双向 | 几何恢复 / 回退 / 双向 | 无关恢复 / 回退 / 双向 |",
                  "|---|---|---:|---:|---:|---:|"])
        for domain in DOMAINS:
            for mode,c in z["domains"][domain].items():
                counts = " | ".join(" / ".join(str(c["counts"][a][k]) for k in ("rescue","damage","dual")) for a in ARMS)
                d.append(f"| {domain} | {NAMES[mode]} | {c['n']} | {counts} |")
        d.extend(["", "下列计数补全真实写入交换后的 K2 输出分布。"])
        for domain in DOMAINS:
            entries = []
            for mode,c in z["domains"][domain].items():
                a = c["counts"]["authentic"]
                other = c["n"] - a["damage_still_correct"] - a["damage"]
                entries.append(f"{NAMES[mode]}保留正答 {a['damage_still_correct']}/{c['n']}，其他结果 {other}/{c['n']}")
            d.append(f"- {domain}。" + "。".join(entries) + "。")
        d.extend(["", "| 格式 | 写入 | 对照 | 多命中 / 少命中 | 净效应〔同时区间〕（百分点） | 单侧上界 / 校正上界（百分点） |",
                  "|---|---|---|---:|---:|---:|"])
        for domain in DOMAINS:
            for mode,c in z["domains"][domain].items():
                for a,x in c["contrasts"].items():
                    lo,hi = [100*y for y in x["simultaneous"]]
                    d.append(f"| {domain} | {NAMES[mode]} | {a} | {x['extra']} / {x['loss']} | {100*x['mean']:+.1f}〔{lo:+.1f}, {hi:+.1f}〕 | {100*x['upper_97_5']:.2f} / {100*x['upper_simultaneous']:.2f} |")
        support = "、".join(NAMES[k] for k in z["supported"]) or "无"
        d.extend(["", f"两种格式均通过双向判据的写入条件为 {support}。同时校正后的最大单侧上界为 {100*z['max_upper']:.2f} 个百分点。预设的区间回放停止条件{'达到' if z['stop'] else '未达到'}。"])
        if z["stop"]:
            d.append("联合门槛之外，所有匹配对照上的额外精确双向命中均被约束在 10 个百分点以内。这个上界只适用于完整分支写入的交换和自然纠正队列。")
        d.extend(["", f"所有受体原生写入回放都恢复了原生成。固定 {z['audit']['n']} 题核验 {z['audit']['checks']} 个注入点。直接重算后缀时末端残差最大偏差 {z['audit']['max_suffix']:.3g}；受体自身写入回放最大偏差 {z['audit']['max_identity']:.3g}。",
                  f"无关行控制中，{v['unrelated_geometry_fallback']:,}/{v['unrelated_token_writes']:,} 个写入位置缺少可缩放的方向，按预设回退到几何控制。此比例纳入对无关行结果的解释。", "",
                  "![两种格式的双向整答](results/MIDDLE_WRITES.png)","",
                  "## 与状态及候选路由证据的关系", "",
                  "第 18 层入口的残差交换曾在冻结受体后续写入时双向改变整答，本轮检验的是更早的完整分支写入在自由重算下能否产生同样的语义变化。单层 Q 的跨表地址转移和双向判据曾失败。两个实验的处理变量与下游条件不同，不能相加为一个中介比例。相关来源为[五边界残差实验](../ouro_lookup_onset_20261121/REPORT_ZH.md)和[单层 Q 实验](../ouro_lookup_query_20261121/REPORT_ZH.md)。"])
        v["figure_sha256"] = plot(z)
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__ == "__main__":
    record = verify()
    render(record)
    save(R/"VERIFICATION.json",record)
    print(record,flush=True)
