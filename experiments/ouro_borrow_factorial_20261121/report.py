"""Verify the exact arithmetic, native token traces, and causal experiment results."""
from collections import Counter
import json
from pathlib import Path
import platform

import torch
import transformers

from borrow import needs
from bidi import unordered
import factorial
import graft
import interaction
import natural
from native import MODEL_HASH, REV, load, save, sha
from rule import PREFIX, parse

ROOT=Path("results")
MODEL="/root/autodl-tmp/ouro-cross-task-20261118/model"


def json_eq(a,b):
    assert json.loads(json.dumps(a))==b


def parsed(r):
    assert r["tokens"] and parse(r["text"])==r["number"]


def baseline_checks():
    key=ROOT/"DECK.json"
    d=load(key)
    actual=load(ROOT/"NATIVE.json")
    assert d["prompt"]==PREFIX and d["previous_sha256"]==sha(factorial.EXTRA)
    assert actual["deck_sha256"]==sha(key) and len(actual["rows"])==len(d["rows"])==200
    seen=factorial.history()
    assert d["excluded_unordered"]==len(seen)
    before=len(seen)
    by={}
    strata=Counter()
    for a,b in zip(d["rows"],actual["rows"],strict=True):
        assert a=={x:y for x,y in b.items() if x!="pred"}
        fields=a["pairs"]
        a0,b0=fields[0]
        a1,b1=fields[3]
        assert fields[1]==[a1,b0] and fields[2]==[a0,b1]
        assert a0//100==a1//100 and b0//100==b1//100
        assert a1-a0==a["a_delta"] and b0-b1==a["b_delta"] and a["a_delta"]+a["b_delta"]==100
        assert [needs(x,y) for x,y in fields]==[False,False,False,True,False]
        assert [y-x for x,y in fields]==a["answers"]
        assert a["answers"]==[a["group"]+100,a["group"]+100-a["a_delta"],a["group"]+a["a_delta"],a["group"],a["group"]]
        assert a["foil"]==a["answers"][0]
        assert all(unordered(z) not in seen for z in fields)
        seen.update(unordered(p) for p in fields)
        assert d["targets"][str(a["group"])]==a["split"]
        assert a["id"] not in by
        strata[a["split"],a["group"]]+=1
        by[a["id"]]=b
        assert len(b["pred"])==2 and all(len(r)==5 for r in b["pred"])
        for r in b["pred"]:
            for p in r:
                parsed(p)
    assert all(x==2 for x in strata.values()) and len(strata)==100 and len(seen)-before==1000
    json_eq(factorial.analyze(actual["rows"],d["targets"]),load(ROOT/"NATIVE_SUMMARY.json"))
    return by


def verify():
    native=baseline_checks()
    gate=load(ROOT/"NATIVE_SUMMARY.json")
    status=dict(status="PASS", model_revision=REV,model_sha256=MODEL_HASH,
                native_sha256=sha(ROOT/"NATIVE.json"),deck_sha256=sha(ROOT/"DECK.json"),
                python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
                cuda=torch.version.cuda,gpu="NVIDIA GeForce RTX 4090",
                math="pinned BF16 promoted to FP32, no TF32, no autocast, no cache",
                complete_int_problem_prompts=1000,independent_target_groups=100)
    p=ROOT/"GRAFT_RAW.json"
    if gate["graft_gate"]:
        g=load(p)
        assert g["native_sha256"]==status["native_sha256"]
        assert [x["id"] for x in g["rows"]]==list(native)
        m=load(ROOT/"GRAFT_SUMMARY.json")
        json_eq(graft.stats(g["rows"]),m)
        tok=transformers.AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True)
        for r in g["rows"]:
            ref=native[r["id"]]
            for i,name in enumerate(("forward","reverse")):
                data=r[name]
                replay=graft.condition(tok,list(native.values()),ref,i)
                assert data["aligned"]==(replay is not None)
                if not data["aligned"]:
                    continue
                case=data["case"]
                assert case=={k:v for k,v in replay.items() if k!="ids"}
                j,a,b=(3,2,1) if i else (0,1,2)
                assert (case["receiver"],case["donors"])==(j,[a,b])
                assert case["expected"]==[ref["answers"][k] for k in (j,a,b,0 if i else 3)]
                assert data["pred"]["native"]==ref["pred"][1][j]
                assert set(case["a"]).isdisjoint(case["b"])
                wrong=native[case["wrong_id"]]
                assert wrong["split"]==ref["split"] and wrong["group"]!=ref["group"]
                assert case["wrong_b"]==wrong["pairs"][1 if i else 2][1]
                for q in data["pred"].values():
                    parsed(q)
        status["graft_sha256"]=sha(p)
        if m["confirmation"]["forward"]["gate"]:
            ic=ROOT/"INTERACTION_RAW.json"
            obj=load(ic)
            assert obj["graft_sha256"]==status["graft_sha256"]
            sample,available=interaction.selection(ROOT,tok,list(native.values()))
            assert [r["id"] for r in obj["rows"]]==[q["id"] for q in sample]
            for r,c in zip(obj["rows"],sample,strict=True):
                assert r["y"]==c["group"] and r["foil"]==c["group"]+100
                assert r["pred"]["joint"]==r["pred"]["restore"]==c["joint_answer"]
                for q in r["pred"].values():
                    parsed(q)
            json_eq(interaction.report(obj["rows"],available),load(ROOT/"INTERACTION_SUMMARY.json"))
            status["interaction_sha256"]=sha(ic)
    if gate["natural_gate"]:
        ar=ROOT/"NATURAL_RAW.json"
        obj=load(ar)
        assert obj["native_sha256"]==status["native_sha256"]
        repair=sorted(x["id"] for x in native.values() if x["split"]=="confirmation" and factorial.corrected(x,3))
        assert [(r["id"],r["kind"]) for r in obj["rows"]]==[(p,s) for p in repair for s in ("repair","safe")]
        for r in obj["rows"]:
            original=native[r["id"]]
            col=3 if r["kind"]=="repair" else 4
            assert r["target"]==original["answers"][col] and r["foil"]==original["foil"]
            assert r["pred"]["native"]==r["pred"]["restore"]==[x[col] for x in original["pred"]]
            for a in r["pred"].values():
                for b in a:
                    parsed(b)
        json_eq(natural.summary(obj["rows"]),load(ROOT/"NATURAL_SUMMARY.json"))
        status["natural_sha256"]=sha(ar)
    save(ROOT/"VERIFICATION.json",status)
    return status


def description(r,rate=False):
    u,v=r["interval"]
    if rate:
        return f"{r['mean']*100:+.2f} [{u*100:+.2f}, {v*100:+.2f}] 个百分点"
    return f"净增 {r['total']} 例，每目标净差 {r['mean']:+.3f} [{u:+.3f}, {v:+.3f}]"


def render():
    native=load(ROOT/"NATIVE_SUMMARY.json")
    top=native["confirmation"]
    n=top["n"]
    d=["# Ouro-1.4B Base 两个低位字段的跨循环组合与自然借位","",
       "固定两操作数各自百位。原题、A 单改和 B 单改都不借位。联合改动跨过借位边界，末两位与原题相同，正确百位减少一。供体的两个单改世界都不是联合题。","",
       "| 原生确认条件 | K1 完整正确 | K2 完整正确 | 精确 +100 错答修复 |","|---|---:|---:|---:|"]
    for p in top["by_corner"]:
        q=f"{p['exact_plus100_recovery']}/{n}" if "exact_plus100_recovery" in p else "不适用"
        d.append(f"| {p['name']} | {p['K1_exact']}/{n} | {p['K2_exact']}/{n} | {q} |")
    d.extend(["",f"原生 K2 四角全对 {top['K2_all_four']}/{n} 组。漏借位原生修复跨 {top['natural_groups']} 个不同目标，配对 {description(top['paired_omission'])}（95% 区间）。"])
    native_description=load(ROOT/"NATIVE_DESCRIPTIVE.json")
    assert native_description["native_sha256"]==sha(ROOT/"NATIVE.json")
    z=native_description["summary"]["confirmation"]
    d.extend([f"在 {z['exact_plus100_repair']} 例原生精确修复中，{z['exact_plus100_after_all_three_singles_correct']} 例的三个无需借位角点在 K1 已经全对。另有 {z['K1_joint_other_wrong_integer']} 个联合角点产生了其他错误整数。"])
    if (ROOT/"GRAFT_SUMMARY.json").exists():
        g=load(ROOT/"GRAFT_SUMMARY.json")["confirmation"]
        d.extend(["","## 独立获取的两个 H1 字段在 K2 中组合","",
                  "正向拼接只使用两个单改题的 H1 隐藏字段，保持受体当前读出及所有未受干预位置。反向以原生联合题为受体。",
                  "", "| 方向 | 原生 | 只换A | 只换B | 两字段联合 | 四角全对 | 等几何四角对照 | 错配B四角对照 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for branch,z in g.items():
            c=z["counts"]
            v=[f"{c.get(q,0)}/{n}" for q in ("native_exact","a_exact","b_exact","ab_exact","quartet","sham_quartet","wrong_b_quartet")]
            d.append("| "+branch+" | "+" | ".join(v)+" |")
        for label,z in g.items():
            d.extend(["",f"{label} 对齐 {z['counts']['aligned']}/{n}，按三种改动幅度的四角通过数与分母为 {z['by_amplitude']}，预设门槛{'通过' if z['gate'] else '未通过'}。"])
            d.extend([f"{label} 中，错配 B 的控制准确生成了与错配数值相符的替代答案 {z['counts'].get('wrong_target_actual',0)}/{n} 次。"])
            for test,result in z["comparison"].items():
                d.extend(["",f"{label} 相对于 {test} 的四角配对差为 {description(result)}（95% 区间）。"])
    if (ROOT/"GRAFT_DESCRIPTIVE.json").exists():
        gdiag=load(ROOT/"GRAFT_DESCRIPTIVE.json")
        assert gdiag["graft_sha256"]==sha(ROOT/"GRAFT_RAW.json")
        sd=gdiag["summary"]["confirmation"]
        d.extend(["",f"描述性分层中，两个供体的原生 K1 都答错时，联合 K2 拼接仍答对 {sd['both_donors_wrong_joint_graft_exact']}/{sd['both_donors_wrong_n']}。受体和两个供体的 K1 都错时，联合 K2 拼接答对 {sd['all_three_wrong_joint_graft_exact']}/{sd['all_three_wrong_n']}。"])
    if (ROOT/"GRAFT_RAW.json").exists():
        native_bank={r["id"]:r for r in load(ROOT/"NATIVE.json")["rows"]}
        chosen=next((r for r in load(ROOT/"GRAFT_RAW.json")["rows"]
                     if r["split"]=="confirmation" and r["amplitude_arm"]!=0 and r["forward"]["aligned"] and
                     [r["forward"]["pred"][k]["number"] for k in ("native","a","b","ab")]==r["forward"]["case"]["expected"]),None)
        if chosen:
            item=native_bank[chosen["id"]]
            a,b=item["pairs"][0]
            a1,b1=item["pairs"][3]
            d.extend(["",f"实例 {chosen['id']} 将 A 从 {a} 改为 {a1}，将 B 从 {b} 改为 {b1}。原题、两个单字段状态干预和联合状态干预的实际输出为 {item['answers'][:4]}。"])
    if (ROOT/"INTERACTION_SUMMARY.json").exists():
        q=load(ROOT/"INTERACTION_SUMMARY.json")
        d.extend(["","## L17 读出中的四角混合差分","",
                  f"在有完整正向四角结果的 {q['eligible']} 个不同目标中抽取 {q['n']} 个。仅在分岔步骤去除第 17 层读出上的 r11−r10−r01+r00，后续仍用原先的两字段 K2 拼接自由续写。","",
                  "| 处理 | 联合正解 | 原始无借位整数 | 其他整数 | 无法解析 |","|---|---:|---:|---:|---:|"])
        for m,v in q["arms"].items():
            d.append("| "+m+" | "+" | ".join(str(v[k]) for k in ("own_joint","base_foil","other","unparsed"))+" |")
        d.extend(["",f"精确回到原始整数的效应为 {description(q['exact_foil_excess'],True)}（95% 区间）。专门回退门槛{'通过' if q['pass_gate'] else '未通过'}。诊断分差见原始数据。"])
    if (ROOT/"NATURAL_SUMMARY.json").exists():
        y=load(ROOT/"NATURAL_SUMMARY.json")
        a=y["repair"]
        nearby=y["safe"]
        d.extend(["","## 原生自然纠错与 L14 字段路径","",
                  "对同一题的受体 K/V，以另一个循环原生查询得到的 AV 差分别回拨两个字段。全查询交换和 AV 结构恢复是单独的对照。",
                  "", "| 处理 | K1 恢复正解 | K2 精确重现旧错答 | 同题双向 | 无需借位对照新增 K2 错误 |",
                  "|---|---:|---:|---:|---:|"])
        for m,v in a["modes"].items():
            d.append(f"| {m} | {v['K1_correct']}/{a['n']} | {v['K2_foil']}/{a['n']} | {v['joint']}/{a['n']} | {nearby['modes'][m]['K2_new_error']}/{nearby['n']} |")
        d.extend(["",f"两条字段分别回拨都恢复同一个旧错答的有 {y['same_cases_both_cuts']} 例，覆盖 {y['both_cut_targets']} 个目标。预设自然路径门槛{'通过' if y['gate'] else '未通过'}。"])
        for m,t in y["rollback"].items():
            d.extend(["",f"{m} 相对几何控制的精确回退为 {description(t)}（97.5% 区间）。"])
    d.extend(["","四角整数、单步读出交互、自然纠错路径的通过条件各自独立。同一受体的字段组合仍可能通过一般数值重算实现。先前无序运算数、各步原生 token 序列和全部统计均有结构与哈希验收。",""])
    Path("REPORT_ZH.md").write_text("\n".join(d),encoding="utf-8")


if __name__=="__main__":
    print(verify(),flush=True)
    render()

