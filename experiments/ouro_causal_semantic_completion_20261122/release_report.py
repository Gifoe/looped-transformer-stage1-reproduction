"""Verify recorded free generations and render the Chinese result report."""

import argparse
from collections import defaultdict
from pathlib import Path

from audit_prior import interpreted_outputs
from csc_data import need, operation, pair_keys, scores
from csc_oracle import RANKS, summary, verify_gate
from native import load, save, sha
from reproduce import MODEL_SHA, ledger_used


def verify(root, checkpoint, gpu):
    assert sha(checkpoint / "model.safetensors") == MODEL_SHA
    deck = load(root / "DECK.json")
    rows = deck["rows"]
    assert len(rows) == 404 and len({r["id"] for r in rows}) == len(rows)
    written, targets = set(), defaultdict(set)
    native = load(root / "NATIVE.json")
    assert native["deck_sha256"] == sha(root / "DECK.json")
    assert len(native["rows"]) == len(rows)
    for expected, actual in zip(rows, native["rows"], strict=True):
        assert all(actual[k] == v for k, v in expected.items())
        targets[expected["form"], expected["target"]].add(expected["split"])
        kind, corners, safe = (expected[k] for k in ("form", "corners", "safe"))
        pairs = pair_keys(corners) | pair_keys(safe)
        assert len(pairs) == 8 and not pairs.intersection(written)
        written.update(pairs)
        assert [need(kind, *p) for p in corners] == [False, False, False, True]
        assert not any(need(kind, *p) for p in safe)
        assert operation(kind, *corners[3]) == operation(kind, *safe[3]) == expected["target"]
    assert all(len(x) == 1 for x in targets.values())
    assert scores(native["rows"]) == load(root / "NATIVE_SUMMARY.json")
    bank = load(root / "BANK_INDEX.json")
    assert bank["native_sha256"] == sha(root / "NATIVE.json") and bank["rows"] == len(rows)
    oracle = load(root / "ORACLE_SUMMARY.json")
    lock = oracle["lock"]
    assert lock == load(root / "ORACLE_LOCK.json")
    assert lock["discovery_sha256"] == sha(root / "ORACLE_DISCOVERY.json")
    assert lock["rank_development_sha256"] == sha(root / "ORACLE_RANK_DEV.json")
    d = {"NATIVE": interpreted_outputs(native)}
    for file, table, splits in (("ORACLE_DISCOVERY", "discovery", ("dev",)),
                                ("ORACLE_RANK_DEV", "rank_development", ("dev",)),
                                ("ORACLE_CONFIRM", "confirmation", ("test", "shift"))):
        raw = load(root / f"{file}.json")
        actual = [(r["id"], r["kind"]) for r in raw["rows"]]
        expected = [(r["id"], k) for r in rows if r["split"] in splits for k in ("joint", "safe")]
        assert actual == expected and oracle[table] == summary(raw["rows"])
        d[file] = interpreted_outputs(raw)
    assert oracle["gate"] == verify_gate(oracle["confirmation"], lock["update"])
    if oracle["gate"]["full"]:
        for k in (1, 2):
            file = f"LEARNED_RAW_K{k}"
            path = root / f"{file}.json"
            if path.exists():
                records = load(path)
                assert [(r["id"], r["kind"]) for r in records["rows"]] == actual
                d[file] = interpreted_outputs(records)
        for file in ("LOOKUP_RAW", "OPERATOR_ABLATION_RAW", "TEMPLATE_RAW", "FORMATION_RAW", "LOOKUP_LATE_RAW"):
            path = root / f"{file}.json"
            if path.exists():
                d[file] = interpreted_outputs(load(path))
    config_files = ("config.json", "configuration_ouro.py", "modeling_ouro.py", "tokenizer.json",
                    "tokenizer_config.json", "merges.txt", "vocab.json", "special_tokens_map.json")
    n = ledger_used(root / "GPU_LEDGER.tsv")
    assert 0 < n <= 18000
    return dict(model_sha256=MODEL_SHA, gpu=gpu, gpu_seconds_including_failures=n,
                outputs_independently_parsed=d, group_targets=len(targets), deck_rows=len(rows),
                distinct_pairs=len(written), previous_pairs_excluded=deck["history_count"],
                historical_inputs_sha256=deck["history_hashes"],
                checkpoint_components={p:sha(checkpoint/p) for p in config_files})


def cell(row, name):
    a = row["arms"][name]
    return f'{a["correct"]}/{a["n"]} (+{a["rescued"]}/−{a["damaged"]})'


def interval(a):
    return f'{100*a["mean"]:+.1f} pp [{100*a["interval"][0]:+.1f}, {100*a["interval"][1]:+.1f}]'


def render(root, v):
    n = load(root / "NATIVE_SUMMARY.json")
    o = load(root / "ORACLE_SUMMARY.json")
    lock, gate = o["lock"], o["gate"]
    winner, rank = lock["update"], lock["chosen_rank"]
    low = f"rank{rank}"
    main = o["confirmation"]["borrow/test/joint"]
    exact = main["arms"][winner]["exact_natural_rescues"]
    ctrl = main["arms"][winner + "_geom"]["exact_natural_rescues"]
    start = ["# Ouro-1.4B 的因果语义补全", "",
             f"本轮使用冻结 Ouro-1.4B Base 和单张 {v['gpu']}。GPU 台账累计 {v['gpu_seconds_including_failures']} 秒，包含首次启动失败。",
             f"开发集选中的局部更新为 `{winner}`。独立借位确认集中，{main['repairs']} 道可被原生 K2 精确纠正的错误中，Oracle 救回 {exact} 道，匹配几何控制救回 {ctrl} 道。对借位确认的联合题和同答案对照题合计改变 {gate['full_gain']:+d} 个正确答案。",
             "", "## 历史证据审计", "",
             "前序操作数四角实验在两处分别替换 K1 字段，K2 的四角整答有 139/140 成功，匹配几何为 0/140。规则切换的 H1 移植可双规则成功 270/288，但回拨原生 V 的精确重现低于匹配对照。旧第 15 层实验在答案分岔处用当轮生成项救回 19/60，并在反向恢复 29/60。由此固定本轮借位、规则和物理切点，并提前至所有答案数字出现之前。 [操作数四角](../ouro_borrow_factorial_20261121/REPORT_ZH.md) · [规则切换](../ouro_rule_switch_20261121/REPORT_ZH.md) · [局部生成项](../ouro_borrow_block_20261121/REPORT_ZH.md)",
             "查表的物理层扫描显示，第 18 层以后两轮间的读出替换明显更强。前序同源扣减的目的行对照在两种题面上超出第三行对照约 63 和 62 个百分点。本轮在查表中固定算术验证过的位置，检验更新和算子可迁移性。这一设定无法代表重新定位的查表专用算子。 [查表出现位置](../ouro_lookup_onset_20261121/REPORT_ZH.md) · [目的行对照](../ouro_lookup_destination_20261122/REPORT_ZH.md)",
             "`PRIOR_AUDIT.json` 另外记录旧语义组织、转移预测的逐题文件和迁移汇总，以及旧 ARC 和配对适配法的逐题成绩、协议及原始记录指纹。旧配对适配法在独立 QASC 确认没有稳定超过联合 CE 对照。 [旧适配法独立审计](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/research/ouro-better-20261120/experiments/ouro_better_20261120)",
             "", "## 冻结数据与协议", "",
             f"冻结 {v['deck_rows']} 组，每组含两个 00/10/01/11 四角，共 {v['distinct_pairs']} 个未重复无序数字对。主四角仅 11 需要跨百位借位或进位，配对四角的 11 答案相同但无需该操作。按规则与目标整数分割，目标组之间没有训练、开发、确认或题面迁移泄漏。生成器排除了 {v['previous_pairs_excluded']:,} 个历史无序输入对。",
             "自由生成使用原生 LM head、贪心解码和无 KV cache 的原生块重算。每步只在末尾提示 token 或已自然生成的连续前导空白写回。无标准数字、字母选项或目标行输入。百分比区间为按目标整数聚类的 10,000 次配对 bootstrap，95% 区间。",
             "", "| 群组 | 联合题 K1 | 联合题 K2 | 同答案对照 K1 | 同答案对照 K2 | 精确原生纠正 |",
             "|---|---:|---:|---:|---:|---:|"]
    for k, row in n.items():
        num = row["n"]
        counts = [row[j][z] for j in ("joint", "safe") for z in ("1", "2")] + [row["repair"]]
        start.append("| " + " | ".join([k, *(f"{x}/{num}" for x in counts)]) + " |")
    start.extend(["", "## 局部更新与 Oracle 干预", "",
                  r"在物理第 15 层，记录原生 Attention/MLP 的实际写入 $W$。`after` 的目标为 $W_{2,11}-W_{1,11}$，包括 K2 进入该层前累积的上游效应。`formed` 则先用 $H_{10}+H_{01}-H_{00}$ 构造 K2 层输入，然后执行真实层并扣去对应四角加性写入。检验 Attention、MLP 及它们的和。若写入 $\Delta$，K1 从该残差节点起继续运行冻结的原生后缀。K2 的相反操作在同一位置扣除该向量。",
                  rf"训练借位组的更新拟合不去中心的正交基 $U$，低秩干预为 $U_r^\top U_r\Delta$。开发集冻结秩 {rank}，{'通过' if lock['rank_dev_gate'] else '未通过'}保留全量效应的开发门槛。全量门槛要求至少 6 道精确救回、至少 10 个独立目标组且相对几何控制区间下限为正，以及联合题与同答案对照净增益为正。",
                  "", "### 开发集选型", "",
                  "| 注入 | 联合题正确（救回/退化） | 原生可纠正样本的救回 | 几何控制救回 | 同答案对照（救回/退化） |",
                  "|---|---:|---:|---:|---:|"])
    for source in (o["discovery"], o["rank_development"]):
        a, b = (source[f"borrow/dev/{k}"] for k in ("joint", "safe"))
        for method in a["arms"]:
            if method + "_geom" not in a["arms"]:
                continue
            x, y = (a["arms"][k]["exact_natural_rescues"] for k in (method, method + "_geom"))
            start.append(f"| `{method}` | {cell(a, method)} | {x}/{a['repairs']} | {y}/{a['repairs']} | {cell(b, method)} |")
    start.extend(["", "### 独立确认与迁移", "",
                  f"借位经典题面的精确纠正子集上，`{winner}` 相比几何控制为 {interval(main['arms'][winner]['repair_vs_geometry'])}。几何臂旋转残差正交分量，保持干预范数与相对原生残差的径向分量。下表的救回与退化以同预算原生 K1 为参照。",
                  "", "| 群组 | 原生 K1 | 全量 Oracle | 几何控制 | 冻结秩 | 冻结秩几何控制 |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name, table in o["confirmation"].items():
        values = [f"{table['native']}/{table['n']}", *(cell(table, arm) for arm in (winner, winner+"_geom", low, low+"_geom"))]
        start.append("| " + " | ".join([name, *values]) + " |")
    start.extend(["", "| 借位确认控制 | 正确（救回/退化） | 原生可纠正样本的救回 |",
                  "|---|---:|---:|"])
    for a in ("wrong_position", "other_operand", *(f"rank{k}" for k in RANKS), *(f"rank{k}_geom" for k in RANKS)):
        start.append(f"| `{a}` | {cell(main,a)} | {main['arms'][a]['exact_natural_rescues']}/{main['repairs']} |")
    paired = o["confirmation"]["borrow/test/safe"]
    start.extend([f"| 同答案供体注入无需借位题 | {cell(paired,'same_answer_donor')} | — |", "",
                  "| 群组 | K2 回滚恢复原错误 | 几何回滚 | 冻结秩回滚 | 冻结秩几何回滚 |",
                  "|---|---:|---:|---:|---:|"])
    for name, arms in o["rollback_exact"].items():
        m = o["confirmation"][name+"/joint"]["repairs"]
        start.append("| " + " | ".join([name, *(f"{arms[k]}/{m}" for k in (winner,winner+"_geom",low,low+"_geom"))]) + " |")
    formation = root / "FORMATION_SUMMARY.json"
    if formation.exists():
        h = load(formation)
        atlas = load(root / "ORACLE_ATLAS.json")["groups"]
        start.extend(["", "### 同剂量比较两轮生成项", "",
                      "在同一第 15 层节点，直接比较 K1 与 K2 的 MLP 四角重放交互。K1 项保留原范数、缩放到 K2 项范数两臂。K2 项另有同剂量几何臂。在首次确认后锁定这些事先未评测的干预，使用同一确认题组，结果视为次级机制证据。",
                      "", "| 群组 | K1 原范数 | K1 与 K2 等范数 | K2 生成项 | K2 几何控制 |",
                      "|---|---:|---:|---:|---:|"])
        for k,v in h["metrics"].items():
            start.append("| " + " | ".join([k,*(cell(v,a) for a in
                         ("formed_k1_natural","formed_k1_equal_norm","formed_k2","formed_k2_geom"))]) + " |")
        for k,v in h["versus_equal_norm_k1"].items():
            q = h["rollback"][k]
            dose = atlas[f"borrow/{k}"]["replay_formed_mlp"]
            blank = atlas[f"borrow/{k}"]["shared_whitespace"]
            start.append(f"\n{k} 的精确纠正子集，K2 项相对等剂量 K1 项为 {interval(v['exact_repair'])}。K2 扣除生成项精确回滚 {q['formed_k2']}/{q['n']}，几何扣减 {q['formed_k2_geom']}/{q['n']}。在提示末尾，K1/K2 中位范数比为 {dose['k1_to_k2']:.3f}。自然共享前导空白后（{blank['with_whitespace']}/{q['n']} 道），同一范数比为 {blank['k1_to_k2']:.3f}，方向中位余弦为 {blank['cosine']:.3f}。")
    start.extend(["", "## 当前轮补全与基准结果", ""])
    phases = []
    for depth in (1,2):
        path = root / f"LEARNED_SUMMARY_K{depth}.json"
        if path.exists():
            phases.append((depth, load(path)))
    if not gate["full"]:
        start.append("Oracle 未达到预注册门槛，当前注入方案停止在因果验证。学习模块、查表评测和 ARC 对照尚未执行。")
    elif not phases:
        start.append("Oracle 达到预注册门槛。学习阶段没有完整的可审计结果，运行日志保留了原因。")
    if phases:
        start.extend([r"算子先将两个操作数末两位、规则位置及上下文与读出位置压到四个 24 维特征。候选为线性、双线性和含门控低秩算子。双线性形式的隐更新为 $W_o[(W_c[z_A,z_B])\odot(W_r[z_R,z_X])]$，经可训练投影写回。训练用因果增量回归、同目标跨表面的功能一致性和真实冻结后缀监督。对照保持相同可训练参数量。Path A 从 K1 第 15 层入口预测写入。Path B 从 K1 的归一化终态预测后写入 K2。",
                      "", "| 路径 | 群组 | 原生 | 语义补全（救回/退化） | 答案监督 Adapter | 状态蒸馏 |",
                      "|---|---|---:|---:|---:|---:|"])
        for depth,s in phases:
            for key,row in s["summary"].items():
                start.append("| " + " | ".join([f"K{depth}",key,f"{row['native']}/{row['n']}",
                         *(cell(row, a) for a in ("causal", "answer_only", "state_distill"))]) + " |")
        for depth,s in phases:
            ms=s["no_cache_batch1_median_ms"]
            stats=s["gate"]
            start.extend(["",f"K{depth} 使用 `{s['arch']}`、秩 {s['rank']}、可训练参数 {s['parameters']:,}。固定 12 题、batch 1 完整生成中位延迟为原生 {ms['native']:.2f} ms、补全 {ms['causal']:.2f} ms、答案监督 {ms['answer_only']:.2f} ms、状态蒸馏 {ms['state_distill']:.2f} ms。",
                          f"借位独立确认中，相对原生 {interval(stats['vs_native'])}，相对答案监督 {interval(stats['vs_ce'])}，相对蒸馏 {interval(stats['vs_distill'])}。题面迁移 {interval(stats['shift'])}，规则迁移 {interval(stats['carry'])}。"])
    baseline = load(root / "PRIOR_AUDIT.json")["arc-pareto-20261119"]
    lookup = root / "LOOKUP_SUMMARY.json"
    if lookup.exists():
        start.extend(["", "### 独立查表", "",
                      "标准和迁移题面各 160 题。冻结算子不经重新拟合。规则位置使用查询键，按展示顺序把奇数行与偶数行的低位分别池化成两个内容输入。系统读入四行，没有标注正确行。Oracle 另测同一层第 15 层的原生 MLP 写入差值。",
                      "", "| 群组 | K1 | K2 | MLP Oracle | 匹配几何 | 算子 K1 | 算子 K2 |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for k,r in load(lookup).items():
            start.append("| " + " | ".join([k,*(cell(r,a) for a in ("native_k1","native_k2","oracle","geometry","causal_k1","causal_k2"))]) + " |")
    late = root / "LOOKUP_LATE_SUMMARY.json"
    if late.exists():
        start.extend(["", "在查表初筛结束后，从每种题面预先生成顺序的前 96 道做第 20 层顺序诊断。使用同题候选无关的真实 $W_2-W_1$，保留第 15 层的同题结果作为位置参照。",
                      "", "| 群组 | K1 | 第15层 MLP | 第20层 Attention | 第20层 Attention 几何 | 第20层 MLP | 第20层 MLP 几何 |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for k,r in load(late).items():
            start.append("| " + " | ".join([k,*(cell(r,a) for a in ("native_k1","oracle","attn_oracle","attn_geometry","mlp_oracle","mlp_geometry"))]) + " |")
            effects = r["vs_geometry"]
            start.append(f"\n{k} 配对几何差，Attention {interval(effects['attn'])}，MLP {interval(effects['mlp'])}。\n")
    diagnosis = root / "OPERATOR_DIAGNOSTIC.json"
    if diagnosis.exists():
        start.extend(["", "### 学习更新与 Oracle 的差距", "",
                      "以下只使用经典借位独立确认集的自然精确纠正题，在无数字前缀的位置测量。相对 RMSE 是整批误差 Frobenius 范数与 Oracle 更新范数之比。",
                      "", "| 路径 | 算子 / 蒸馏中位余弦 | 更新范数比中位数 | 相对 RMSE | 算子精确救回 | 整答原算子 | 互换操作数输入 | 消除两操作数输入 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for key, diag in load(diagnosis).items():
            p = diag["patch_geometry"]
            distill = diag["distill_geometry"]
            repairs = diag["exact_repair_outputs"]
            raw = diag["free_results"]
            start.append(f"| {key} | {p['cosine_median']:.3f} / {distill['cosine_median']:.3f} | {p['norm_ratio_median']:.3f} | {p['relative_rmse']:.3f} | "
                         f"{repairs['unmodified']}/{repairs['n']} | "
                         f"{raw['unmodified']}/{raw['n']} | {raw['swap']}/{raw['n']} | {raw['no_content']}/{raw['n']} |")
    template = root / "TEMPLATE_SUMMARY.json"
    if template.exists():
        s = load(template)
        a = s["settings"]
        start.extend(["", "### 秩一方向与门控解耦", "",
                      f"候选方向为训练组真纠正写入的均值，以及它与同答案安全写入的配对差均值。使用固定方向乘 K1 隐状态双线性门控，含 {s['parameters']} 个可训练参数。开发集选中 `{a['direction']}`、强度 {a['gain']}。完整生成中位延迟为 {s['no_cache_batch1_median_ms']:.2f} ms。",
                      "Oracle 门控只在离线诊断中告知算法原生 K2 能否精确纠正。恒定门控采用训练集预测幅度均值。",
                      "", "| 群组 | 原生 K1 | K1 隐状态门控 | 恒定门控 | Oracle 门控 | 初始 K1 算子 |",
                      "|---|---:|---:|---:|---:|---:|"])
        for k,t in s["summary"].items():
            start.append("| " + " | ".join([k,f"{t['native']}/{t['n']}",*(cell(t,p) for p in
                         ("causal","constant_gate","oracle_gate","previous_causal"))]) + " |")
        start.append("\n借位确认中，秩一算子对恒定门控的配对差为 " + interval(s["versus_constant"]) + "。")
        for k,q in s["selector"].items():
            start.append(f"{k} 档在空生成前缀识别原生精确纠正事件的 AUROC 为 {q['auc']:.3f}，其中阳性 {q['positive']}，阴性 {q['negative']}。")
    start.extend(["",f"ARC-Challenge 历史实验已使用全量 Test 的 {baseline['n']:,} 道题，按完整候选内容打分。本轮对逐题候选分数重新核算 `acc` 和按字符数归一的 `acc_norm`。这些数据来自历史对照。",
                  "", "| 深度 | 原生 acc / acc_norm | 历史开发集锁定对照及其 Test 分数 | LoopCD 各变体在 Test 的分数上限（acc / acc_norm） |",
                  "|---:|---:|---:|---:|"])
    for k in (1,2,3,4):
        def score(name):
            return "/".join(f"{100*baseline[f'verified_{a}_numerators'][name]/baseline['n']:.2f}%"
                            for a in ("acc","acc_norm"))
        selected=baseline["development_selected"][str(k)]
        arms=[n for n in baseline["variants"] if n.startswith("loopcd_") and n.endswith(str(k))]
        top="/".join(f"{100*max(baseline[f'verified_{a}_numerators'][n] for n in arms)/baseline['n']:.2f}%"
                     for a in ("acc","acc_norm")) if arms else "—"
        start.append(f"| K{k} | {score(f'native_k{k}')} | `{selected}` {score(selected)} | {top} |")
    start.append("")
    return start, phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("gpu")
    args = parser.parse_args()
    verification = verify(args.root, args.checkpoint, args.gpu)
    save(args.root / "VERIFICATION.json", verification)
    text, _ = render(args.root, verification)
    (args.root.parent / "REPORT_ZH.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()
