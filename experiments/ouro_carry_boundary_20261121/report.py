"""Build a report and validate the release against the frozen deck."""
from collections import Counter
from pathlib import Path
import platform

import torch
import transformers

from analysis import estimate, repaired
from boundary import CONDITIONS, scores
from carry import PROMPT, numeric
from native import MODEL_HASH, REV, load, save, sha


def validate(root, previous):
    native = load(root/"NATIVE.json")
    locked = load(root/"MANIFEST.json")
    gate = load(root/"BEHAVIOR_GATE.json")
    assert native["manifest_sha256"] == sha(root/"MANIFEST.json")
    assert gate["native_sha256"] == sha(root/"NATIVE.json")
    assert locked["previous_sha256"] == sha(previous)
    assert len(native["rows"]) == len(locked["rows"]) == 288
    assert locked["prompt"] == PROMPT
    empty = set()
    mirrors_total = Counter()
    old = {tuple(sorted(p)) for x in load(previous)["rows"] for p in x["pairs"]}
    by_group = []
    for m, r in zip(locked["rows"],native["rows"],strict=True):
        assert (m["id"],m["split"]) == (r["id"],r["split"])
        duplicate = False
        for c in CONDITIONS:
            a = m["conditions"][c]
            b = r["conditions"][c]
            key = tuple(a["pair"])
            assert a == {k:v for k,v in b.items() if k != "outputs"}
            assert key not in empty and a["target"] == sum(key)
            assert all(numeric(o["text"]) == o["number"] and o["tokens"]
                       for o in b["outputs"])
            old_mirror = tuple(sorted(key)) in old
            within_mirror = tuple(reversed(key)) in empty
            if old_mirror or within_mirror:
                mirrors_total[r["split"]]+=1
            if c in ("stop","same_stop_no_incoming"):
                duplicate = duplicate or old_mirror or within_mirror
            empty.add(key)
        if r["split"]=="confirmation" and not duplicate:
            g=r["conditions"]
            by_group.append(int(repaired(g["stop"]))-int(repaired(g["same_stop_no_incoming"])))
    assert scores(native["rows"]) == load(root/"NATIVE_SUMMARY.json")
    fix = load(root/"NATIVE_SUMMARY.json")["confirmation"]
    assert gate["primary_repaired"] == fix["stop"]["foil_repaired"]
    assert gate["primary_minus_matched"]["total"] == fix["stop"]["foil_repaired"] - fix["same_stop_no_incoming"]["foil_repaired"]
    assert gate["pass"] == (gate["primary_repaired"] >= 16 and gate["primary_minus_matched"]["interval"][0] > 0)
    body=dict(status="PASS", model_revision=REV, model_weights_sha256=MODEL_HASH,
              python=platform.python_version(), torch=torch.__version__,
              transformers=transformers.__version__, cuda=torch.version.cuda,
              gpu="NVIDIA GeForce RTX 4090", model_math="bf16 checkpoint promoted to FP32; TF32 disabled; use_cache=False; no autocast",
              manifest_sha256=sha(root/"MANIFEST.json"), native_sha256=sha(root/"NATIVE.json"),
              ordered_unique_count=len(empty), unordered_overlap_prompt_count=dict(mirrors_total),
              sensitivity_no_reused_unordered_primary_pair=estimate(by_group))
    path = root/"SELECTION.json"
    if path.exists():
        sel=load(path)
        assert sel["raw_sha256"]==sha(root/"DISCOVERY_RAW.json")
        assert sel["causal_protocol_sha256"]==sha(Path("CAUSAL_PROTOCOL.md"))
        body["selected_branches"]=[f"L{q['layer']:02}_{q['write']}" for q in sel["winners"]]
        for site in body["selected_branches"]:
            detail=load(root/f"CONFIRMATION_RAW_{site}.json")
            assert detail["selection_sha256"]==sha(path) and detail["native_sha256"]==sha(root/"NATIVE.json")
            assert len(detail["rows"])==192
        summary=load(root/"CAUSAL_SUMMARY.json") if body["selected_branches"] else {}
        assert set(body["selected_branches"])==set(summary)
    save(root/"VERIFICATION.json",body)
    return body


def percentage(v):
    return f"{v*100:+.2f}"


def effect(x):
    a,b=x["interval"]
    return f"{percentage(x['mean'])} pp [{percentage(a)}, {percentage(b)}]"


def render(root, audit):
    b=load(root/"NATIVE_SUMMARY.json")["confirmation"]
    g=load(root/"BEHAVIOR_GATE.json")
    doc=[
        "# Ouro-1.4B Base 十位进位边界与跨循环写入干预",
        "",
        "候选自由生成。固定同一组六个整数加法示例，无答案列表。贪心生成完整词表，按第一个非空行的完整整数判分。权重冻结。K1 和 K2 除循环次数外使用相同解码设置。",
        "",
        "## 独立题组的行为复现",
        "",
        "每个条件有 192 个确认组。另有 96 个独立发现组用于选择物理支路。",
        "",
        "| 条件 | K1 正确 | K2 正确 | K1 精确错值 | K2 精确错值 | 指定错值→正确 | 所有纠正 | 回退 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    names=dict(stop="应停止",continue_="应继续",safe="远离边界",same_stop_no_incoming="同答案且无个位进位（应停止对照）",
               same_continue_no_incoming="同答案且无个位进位（应继续对照）",neighbor="边界邻近尾数")
    for condition,q in b.items():
        cols=[q[n] for n in ("correct_K1","correct_K2","foil_K1","foil_K2","foil_repaired","WC","CW")]
        doc.append("| "+str(names.get(condition, names.get(condition+"_")))+" | "+" | ".join(map(str,cols))+" |")
    phrase="通过" if g["pass"] else "未通过"
    s=audit["sensitivity_no_reused_unordered_primary_pair"]
    doc.extend(["",f"主终点为 K1 恰好多 100、K2 完整整数正确。结果 **{g['primary_repaired']}/192**，与完全同答案无入进位对照的配对差为 **{effect(g['primary_minus_matched'])}**（95% 区间），{phrase}预设门槛。应继续条件的精确错值为少 100，其余停止相关条件为多 100。",
       "",f"数据锁保证运算数有序对唯一。另排除与先前试验或本轮先出现的题互为运算数调换的主终点组，剩 {s['n']} 组，配对差为 {effect(s)}。所有点估计及区间以题组为单位。",
       "", "## 原生残差写入的因果干预", ""])
    path=root/"CAUSAL_SUMMARY.json"
    if path.exists():
        selected=load(root/"SELECTION.json")
        stats=load(path)
        doc.extend([
            "对状态轨迹中分支写入做向量替换，在原提示最后一个 token 干预。自然剂量为同一分支的 K2 写入减 K1 写入。K1 加它，K2 减它。对照使用相同位置、相同范数的高斯方向，并检验来自不同正确百位的自然供体方向。在独立确认组重新自由生成。零剂量路径逐题复现两种深度的原生 token 序列。",
            "",
            "| 支路 | 发现差（logit） | K1 恢复 自然/等范数/供体 | K2 恢复 +100 错值 自然/等范数 | K1 超额恢复 | K2 超额恢复错值 | 供体超额恢复 |",
            "|---|---:|---:|---:|---|---|---|"])
        for site in selected["winners"]:
            key=f"L{site['layer']:02}_{site['write']}"
            p=stats[key]["stop"]
            n=p["primary_repaired"]
            c=[p[x+"_correct_in_primary"] for x in ("k1_swap","k1_noise","k1_other")]
            w=[p[x+"_foil_in_primary"] for x in ("k2_reverse","k2_noise")]
            f,r,t=[p["primary_all_groups"][m] for m in ("forward","reverse","transfer")]
            doc.append(f"| {key} | {site['score']:.3f} | {' / '.join(map(str,c))}（{n} 题） | {' / '.join(map(str,w))}（{n} 题） | {effect(f)} | {effect(r)} | {effect(t)} |")
        doc.extend(["","计数限定为对应原生错值修复题。配对差以全部 192 组为分母，其他组记零。表中区间为 97.5%。继续进位、同整数且无入进位对照及无条件结果见 CAUSAL_SUMMARY.json。"])
        for site,conditions in stats.items():
            p=conditions["stop"]
            claims=p["primary_all_groups"]
            local=all(claims[m]["interval"][0]>0 for m in ("forward","reverse"))
            transferable=local and claims["transfer"]["interval"][0]>0
            doc.extend(["",f"{site}：双向局部中介门槛{'通过' if local else '未通过'}，供体超额恢复及双向门槛{'同时通过' if transferable else '未同时通过'}。继续进位干预中的 K1 完整正确量为 {conditions['continue']['correct']['k1_swap']}/192，原生为 {conditions['continue']['correct']['k1_native']}/192。受体 stop 条件生成供体完整整数的有 {p['donor_integer_transfer']}/192。"])
        doc.extend(["","这些干预直接改变一个分支的残差写入。内部实现可能同时改变多种特征。即使两向干预成立，结论仍限定于所测检查点、提示、长度和进位边界。跨循环供体使用是离线诊断。"])
    elif g["pass"]:
        doc.append("发现集未选出满足预设标准的物理支路。")
    else:
        doc.append("根据行为门槛停止支路实验。应停止与同答案对照均出现百位过高后修复。在相同答案下，当前结果无法把这种修复定位到入进位的停止计算。尾数 93/95 的邻近组也出现同类错误，因此结果同样不足以把修复归因于精确尾数 94。")
    doc.extend(["", "## 复现", "", "PREREGISTRATION.md 和 CAUSAL_PROTOCOL.md 记录预设门槛。VERIFICATION.json 记录精度、模型哈希和数据校验。逐题记录包含全部完整生成与 token ID。环境命令见 run.sh 和 run_causal.sh。", ""])
    Path("REPORT_ZH.md").write_text("\n".join(doc),encoding="utf-8")


if __name__ == "__main__":
    root=Path("results")
    metadata=validate(root,Path("/root/ouro-open-carry-20261121/MANIFEST.json"))
    render(root,metadata)

