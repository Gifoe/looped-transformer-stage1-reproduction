"""Verify leave-out arithmetic, full generated outputs and causal tests."""
import json
import os
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import transformers

from bidi import unordered
from borrow import needs
from edge_deck import PARENT, history
from edge_stats import ARMS, summarize
from native import MODEL_HASH, REV, load, save, sha
from native_deck import is_repair, report as report_native
from rule import PREFIX, parse

R=Path("results")


def check(pred):
    assert pred["tokens"] and pred["number"]==parse(pred["text"])


def equals(left,right):
    assert json.loads(json.dumps(left))==right


def validate(root,primary=None):
    deck=load(root/"DECK.json")
    nat=load(root/"NATIVE.json")
    seen=history()
    if primary is None:
        assert deck["parent_sha256"]==sha(PARENT)
    else:
        main=load(primary/"DECK.json")
        assert deck["primary_deck_sha256"]==sha(primary/"DECK.json")
        assert [r for r in deck["rows"] if r["split"]=="discovery"]==[r for r in main["rows"] if r["split"]=="discovery"]
        assert all(y not in main["groups"] for y,s in deck["groups"].items() if s=="confirmation")
        for r in main["rows"]:
            seen.update(unordered(x) for k in ("borrow","safe") for x in r[k])
    assert deck["earlier_unordered"]==len(seen)
    assert nat["deck_sha256"]==sha(root/"DECK.json") and deck["prompt"]==PREFIX
    by={}
    for a,b in zip(deck["rows"],nat["rows"],strict=True):
        assert a=={k:v for k,v in b.items() if k!="pred"} and a["id"] not in by
        assert a["target"]==a["foil"]-100 and deck["groups"][str(a["target"])]==a["split"]
        by[a["id"]]=b
        for kind in ("borrow","safe"):
            q=a[kind]
            a0,b0=q[0]
            a1,b1=q[3]
            assert q[1:3]==[[a1,b0],[a0,b1]]
            assert [a1-a0,b0-b1]==a[kind+"_steps"]
            assert all(len(str(x))==a["width"] for cell in q for x in cell)
            assert [needs(*p) for p in q]==[False,False,False,kind=="borrow"]
            assert b1-a1==a["target"] and (kind!="borrow" or b0-a0==a["foil"])
            for p in q:
                u=unordered(p)
                if primary is None or a["split"]=="confirmation":
                    assert u not in seen
                    seen.add(u)
                else:
                    assert u in seen
        assert [len(q) for q in b["pred"]]==[5,5]
        for depth in b["pred"]:
            for answer in depth:
                check(answer)
    assert len(by)==(280 if primary is None else 140)
    assert len(seen)-deck["earlier_unordered"]==8*(len(by) if primary is None else len(by)-40)
    if primary is not None:
        old=load(primary/"NATIVE.json")["rows"]
        assert [r for r in nat["rows"] if r["split"]=="discovery"]==[r for r in old if r["split"]=="discovery"]
    original=load(root/"NATIVE_SUMMARY.json")
    equals(report_native(nat["rows"],deck["groups"]),original)
    verified=dict(status="PASS",model_revision=REV,model_weights_sha256=MODEL_HASH,
                  actual_weights_sha256=os.environ["OURO_ACTUAL_SHA256"],gpu=os.environ["OURO_GPU"],
                  pytorch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
                  deck_sha256=sha(root/"DECK.json"),native_sha256=sha(root/"NATIVE.json"),
                  scored_prompts=5*(len(by) if primary is None else len(by)-40))
    assert MODEL_HASH==verified["actual_weights_sha256"]
    if primary is None and not original["gate"]:
        return verified
    lock=load(root/"HEAD_LOCK.json")
    if primary is not None:
        old=load(primary/"HEAD_LOCK.json")
        assert lock["primary_lock_sha256"]==sha(primary/"HEAD_LOCK.json")
        assert {k:v for k,v in lock.items() if k not in ("primary_lock_sha256","deck_sha256")}=={
                k:v for k,v in old.items() if k!="deck_sha256"}
    assert lock["deck_sha256"]==verified["deck_sha256"] and len(lock["evaluated"])==6
    for c in lock["evaluated"]:
        score=(torch.tensor(c["target"])-torch.tensor(c["control"])).mean(dim=-1).median()
        assert score.item()==c["score"]
    winner=max(lock["evaluated"],key=lambda z:z["score"])
    assert (winner["layer"],winner["head"])==(lock["layer"],lock["head"])
    assert set(lock["focus"])<=set(lock["dev"]) and all(by[r]["split"]=="discovery" for r in lock["dev"])
    raw=load(root/"EDGE_RAW.json")
    assert raw["native_sha256"]==verified["native_sha256"] and raw["lock_sha256"]==sha(root/"HEAD_LOCK.json")
    expected=sorted(x["id"] for x in nat["rows"] if x["split"]=="confirmation" and is_repair(x))
    assert [(r["id"],r["kind"]) for r in raw["rows"]]==[(x,k) for x in expected for k in ("repair","safe")]
    for r in raw["rows"]:
        s=by[r["id"]]
        assert all(r[k]==s[k] for k in ("width","target","foil"))
        ix=3 if r["kind"]=="repair" else 4
        assert r["native"]==[q[ix] for q in s["pred"]]
        assert set(r["pred"])==set(ARMS) and r["pred"]["native"]==r["pred"]["self_restore"]==r["native"][1]
        for a in r["pred"].values():
            check(a)
        if r["aligned"]:
            donor=by[r["donor"]["id"]]
            assert donor["split"]=="discovery" and r["width"]==donor["width"]
            assert all(donor[k]//100!=r[k]//100 for k in ("target","foil"))
            for k in ("mass","matched_mass"):
                assert len(r[k])==4 and all(0<=x<1 for x in r[k])
            assert all(a<=b+1e-7 for a,b in zip(r["matched_mass"],r["mass"],strict=True))
        else:
            assert all(a==r["native"][1] for a in r["pred"].values())
    equals(summarize(raw["rows"]),load(root/"EDGE_SUMMARY.json"))
    verified.update(raw_sha256=sha(R/"EDGE_RAW.json"),head_lock_sha256=raw["lock_sha256"],
                    runtime_s=raw["elapsed"],peak_allocated_mib=raw["peak_allocated_mib"])
    return verified


def paired(v):
    a,b=v["interval"]
    if v["rate"] is None:
        return "无样本"
    return f"{v['total']:+d} 道，{100*v['rate']:+.1f} [{100*a:+.1f}, {100*b:+.1f}] 个百分点"


def figure(z,raw):
    fig,axs=plt.subplots(1,2,figsize=(9.4,3.6))
    colors=("#365D84","#977155","#929DA7")
    counts=z["repair"]["counts"]
    arms=("readcut","matched_read","geometry")
    v=axs[0].bar(range(3),[counts[k]["foil"] for k in arms],color=colors,width=.55)
    axs[0].bar_label(v,padding=2,fontsize=9)
    axs[0].set_xticks(range(3),["Operand edge","Unrelated edge","Write geometry"],rotation=12)
    axs[0].set_ylim(0,max(3,axs[0].get_ylim()[1]*1.14))
    axs[0].set_ylabel("Exact +100 reversions")
    axs[0].set_title("K2 full integers after K1 cuts",loc="left")
    keep=[r for r in raw if r["kind"]=="repair" and r["aligned"]]
    data=[[r["margin"][k]-r["margin"]["native"] for r in keep] for k in arms]
    if keep:
        axs[1].boxplot(data,showfliers=False)
        axs[1].set_xticks([1,2,3],["Operand edge","Unrelated edge","Write geometry"])
    axs[1].axhline(0,color="#929DA7",lw=.8)
    axs[1].set_ylabel("Change in target-minus-foil logit")
    axs[1].set_title("Diagnostic readout margin",loc="left")
    axs[1].tick_params(axis="x",labelrotation=12)
    for ax in axs:
        for s in ("top","right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
    fig.tight_layout()
    path=R/"K1_EDGE_K2.png"
    fig.savefig(path,dpi=170,bbox_inches="tight")
    plt.close(fig)
    with Image.open(path) as im:
        assert im.width>=1000 and im.height>=420
        im.verify()
    return sha(path)


def render(verified):
    original=load(R/"NATIVE_SUMMARY.json")
    q=original["confirmation"]
    nat=load(R/"NATIVE.json")["rows"]
    d=["# K1 操作数注意力读边与 K2 借位修复","",
       "本轮在冻结的 Ouro-1.4B Base 中，一次只切一个物理头在 B 末位查询 A 低两位的两条读边。保持原生 V、输出投影、归一化与后续算子。对删去概率后的其余键按原注意力比例归一。所有决定性指标均为自由续写的完整整数。","",
       f"240 个确认题干中，借位联合角由 +100 精确转为正解 {q['total']['joint_repairs']}/240，同答案的无需借位对照为 {q['total']['matched_safe_repairs']}/240。","",
       "| 操作数宽度 | n | 借位精确修复 | 无借位精确修复 | 借位 K1 正解 | 借位 K2 正解 |",
       "|---|---:|---:|---:|---:|---:|"]
    for w in (3,4):
        v=[r for r in nat if r["split"]=="confirmation" and r["width"]==w]
        d.append(f"| {w} 位 | {len(v)} | {sum(map(is_repair,v))} | "
                 f"{sum([p[4]['number'] for p in r['pred']]==[r['foil'],r['target']] for r in v)} | "
                 f"{sum(r['pred'][0][3]['number']==r['target'] for r in v)} | "
                 f"{sum(r['pred'][1][3]['number']==r['target'] for r in v)} |")
    if not original["gate"]:
        d.extend(["","原生门槛未通过，按预设规则停止因果干预。"])
    else:
        win=load(R/"HEAD_LOCK.json")
        z=load(R/"EDGE_SUMMARY.json")
        raw=load(R/"EDGE_RAW.json")["rows"]
        d.extend(["",f"开发集从全部 {win['screen_shape'][0]} 层 × {win['screen_shape'][1]} 头筛选六个候选，用真实 K2 传播和生成变化锁定物理第 {win['layer']+1} 层第 {win['head']+1} 头，选择代码不读取自然答案。独立确认候选 {z['repair']['n']} 个，读出对齐 {z['repair']['aligned']} 个。","",
                  "| 条件 | 借位正解 | 借位 +100 | 同题回退后复原 | 无借位新增错误 |",
                  "|---|---:|---:|---:|---:|"])
        for arm in ARMS:
            a,b=z["repair"]["counts"][arm],z["safe"]["counts"][arm]
            d.append(f"| {arm} | {a['correct']} | {a['foil']} | {a['restored']} | {b['harm']} |")
        d.extend(["","## 锁定规则下的判定","",
                  f"读边支持门槛{'通过' if z['gates']['read_edge'] else '未通过'}。联合复位与异题供体的完整链门槛{'通过' if z['gates']['full_chain'] else '未通过'}。10% 效应规模排除门槛{'通过' if z['gates']['exclude_ten_percent'] else '未通过'}。"])
        for kind,vals in z["strata"].items():
            d.extend(["",f"### {kind}（{vals['n']} 道）",""])
            for k,c in vals["contrasts"].items():
                d.append(f"- {k} 为 {paired(c)}（配对 97.5% 区间）。")
            for k,c in vals["upper_bounds"].items():
                d.append(f"- 相对 {k}，{c['positive_targets']}/{c['total_targets']} 个答案簇出现至少一次额外回退，单侧 97.5% 上界为 {100*c['upper97_5']:.2f}%。")
        a=z["repair"]["median"]
        if a:
            d.extend(["",f"原生联合角切除概率质量中位数 {a['true_mass']:.4f}，无关边对照 {a['matched_mass']:.4f}。成功配平到绝对差 ≤0.001 的题占 {100*a['control_fraction_within_001']:.1f}%。切边造成 K2 第 15 层传播项和 MLP 生成项的范数变化中位数分别为 {a['propagation_shift']:.4f} 和 {a['new_mlp_shift']:.4f}，原生对应分量范数为 {a['K2_native_P']:.4f} 和 {a['K2_native_M']:.4f}。"])
            donor=z["repair"]["counts"]["donor_borrow"]
            d.extend(["",f"借位供体干预后直接输出供体真值、供体原错各 {donor['donor_exact']}、{donor['donor_foil']} 道，仅按输出百位比对分别为 {donor['donor_target_digit']}、{donor['donor_foil_digit']} 道。供体选择不共享真值百位或原错百位。"])
        d.extend(["","逐题边质量、完整答案、供体编号、原生分差和切边后的三个 K2 项全部保留。效应上界约束这一个事先锁定的头和读边，适用人群由产生预设精确修复的宽度决定。","",
                  "![操作数读边和 K2 结果](results/K1_EDGE_K2.png)"])
        verified["figure_sha256"]=figure(z,raw)
    ext=R/"extension"
    if (ext/"EDGE_SUMMARY.json").exists():
        second=load(ext/"EDGE_SUMMARY.json")
        new=load(ext/"EDGE_RAW.json")["rows"]
        initial=[r for r in raw if r["width"]==3]
        assert {r["target"] for r in initial}.isdisjoint({r["target"] for r in new})
        complete=summarize(initial+new)
        save(R/"THREE_DIGIT_COMBINED.json",complete)
        verified["three_digit_summary_sha256"]=sha(R/"THREE_DIGIT_COMBINED.json")
        d.extend(["","## 独立三位数增补","",
                  "主确认集只有 34 个独立精确修复簇。在读取干预结果前，仅据原生样本量固定 100 道额外题。增补照用同一头、供体池和阈值，答案簇和主队列互不重叠。四位数没有符合预设的原生精确修复，因此未进入该合并估计。","",
                  "| 分组 | 原生自然修复 | 切读边精确回退 | 无关读边精确回退 | 写入几何精确回退 |",
                  "|---|---:|---:|---:|---:|"])
        for label,s in (("增补三位",second),("合并三位",complete)):
            r=s["repair"]
            counts=[r["counts"][a]["foil"] for a in ("readcut","matched_read","geometry")]
            d.append(f"| {label} | {r['n']} | {counts[0]} | {counts[1]} | {counts[2]} |")
        for k,v in complete["strata"]["pooled"]["upper_bounds"].items():
            d.extend(["",f"合并后相对 {k} 的额外回退出现于 {v['positive_targets']}/{v['total_targets']} 个独立答案簇，单侧 97.5% 上界 {100*v['upper97_5']:.2f}%。"])
        d.extend(["",f"合并三位队列的 10% 效应排除门槛{'通过' if complete['gates']['exclude_ten_percent'] else '未通过'}。"])
    Path("REPORT_ZH.md").write_text("\n".join(d)+"\n",encoding="utf-8")


if __name__=="__main__":
    v=validate(R)
    if (R/"extension"/"EDGE_SUMMARY.json").exists():
        v["extension"]=validate(R/"extension",primary=R)
    render(v)
    save(R/"VERIFICATION.json",v)
    print(v,flush=True)
