"""Assemble the RELMT + CFIT report from distinct historical, prospective, and official evidence."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def put(path,text):
    staging=path.with_suffix(".partial")
    staging.write_text(text,encoding="utf-8")
    staging.replace(path)


def coeff(entry):
    point=entry["estimate"]
    ci=entry["ci95"]
    return "—" if point is None else (f"{point:+.3f}" if ci is None else f"{point:+.3f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")


def report(v4,official):
    label=official/"SELECTION_V4.json"
    token=sha(label)
    selection=read(label)
    table_file=official/"official/FINAL_8PLUS4_TABLE.json"
    paired_file=table_file.with_name("PAIRED_SIGNIFICANCE.json")
    table=read(table_file)
    paired=read(paired_file)
    if (token!=label.with_suffix(".sha256").read_text().split()[0] or
        any(output["selection_sha256"]!=token for output in (table,paired)) or
        table["total_count"]!=57379 or table["historical_count"]!=43769):
        raise ValueError("The official evaluation is not complete for the signed candidates")
    evidence={
        "T1":v4/"exp_t1/T1_SIGMA_METRIC_AUDIT.json",
        "T2-A":v4/"exp_t2a_self_distill/T2A_SELF_DISTILL.json",
        "T2-B":v4/"exp_t2b_prose_contrast/T2B_PROSE_CONTRAST.json",
        "T3-A":v4/"exp_t3a/T3A_PROTOCOL_INVERSION.json",
        "T4 forecast":v4/"prospective_unseen/FORECAST_LOCK_V4_PROSPECTIVE.json",
        "T4 measured":v4/"prospective_unseen/PROSPECTIVE_UNSEEN11_STRICT_AUDIT.json",
        "T5 recipe":v4/"handoff_codex/T5_TRAINING_LOCK.json",
        "T5 capacity":v4/"exp_t5_mixing/CAPACITY_CERTIFICATE.json",
        "Non-ARC filter":v4/"handoff_codex/T5_NONARC_AUDIT.json",
        "Selection":label,
        "Official totals":table_file,
        "Official pairs":paired_file,
    }
    v={name:read(path) for name,path in evidence.items()}
    t1=v["T1"]["spectrum"]
    t2=v["T2-A"]
    lines=[
        "# Ouro-1.4B RELMT + CFIT 实验报告", "",
        "## T1　接收状态几何", "",
        f"无标签语料的前 16 个主成分解释 {100*t1['unlab_top16_frac']:.2f}% 的接收状态方差，"
        f"候选语料上为 {100*t1['cand_top16_frac']:.2f}%。"
        "协方差加权相似度描述算子对已测状态的作用，性能预测还需独立行为数据。","",
        "## T2　零标签协议自蒸馏与散文对比","",
        "| 伪标签训练源 | 两种子开发 Macro5 增益 | Cloze3 增益 |",
        "| --- | ---: | ---: |",
    ]
    for data in ("arc","csqa","mmlu"):
        runs=[s for name,s in t2["arms"].items() if name.startswith(f"self_{data}_hardpseudo.")]
        if len(runs)!=2:
            raise ValueError("The completed two-seed hard-pseudo policy is required")
        lines.append(f"| {data} | {sum(s['delta_macro5_pp'] for s in runs)/2:+.2f} pp | {sum(s['delta_cloze3_pp'] for s in runs)/2:+.2f} pp |")
    lines.extend(["",f"正式评测沿用此前按纯无标签留出集一致率锁定的控制器，{t2['frozen_self_anchor']['arm']}。"
                     "显示选项与 Cloze 的差异是两种读取协议的行为证据。","",
                  "| 散文信赖域系数 | 种子数 | 平均开发 Macro5 增益 |","| ---: | ---: | ---: |"])
    for lam in (0,1,3):
        arms=[a for key,a in v["T2-B"]["arms"].items() if key.startswith(f"prose_rank_lam{lam}.")]
        if len(arms)!=2:
            raise ValueError("Missing paired natural-text runs")
        lines.append(f"| {lam} | 2 | {sum(a['delta_macro5_pp'] for a in arms)/2:+.2f} pp |")
    lines.extend(["","## T3　同题协议与源内对照","",
                  "| 训练协议 | 两种子 Macro5 | 两种子 Cloze3 |",
                  "| --- | ---: | ---: |"])
    for mode,definition in v["T3-A"]["summary"].items():
        lines.append(f"| {mode} | {definition['delta_macro5_pp']:+.2f} pp | {definition['delta_cloze3_pp']:+.2f} pp |")
    lines.extend(["","| 来源 | Regime III | Regime II | III − II |",
                  "| --- | ---: | ---: | ---: |"])
    for source in ("medmcqa","qasc","obqa","race"):
        p=v4/"exp_t3b_reversal"/f"T3B_{source}.json"
        evidence[f"T3-B {source}"]=p
        stats=read(p)["summary"]
        a,b=(stats[name]["delta_macro5_pp"] for name in ("Top_RegimeIII","Top_RegimeII"))
        lines.append(f"| {source} | {a:+.2f} pp | {b:+.2f} pp | {a-b:+.2f} pp |")
    lines.extend(["","各来源的组别通过冻结信号选取等量题目，并沿用相同训练预算。报告保留四个来源及其负结果。","",
                  "## T4　锁定的 11 源前瞻检验","",
                  "旧 14 源实测在理论开发时已经可见，列作回顾性诊断。下表的 11 源在训练前冻结预测与数据池。区间通过来源重采样估计。","",
                  "| 预测器 | Spearman ρ 与 95% CI | Kendall τ 与 95% CI | 前三命中 | Top-1 regret |",
                  "| --- | --- | --- | ---: | ---: |"])
    for metric,item in v["T4 measured"]["predictor_audit"].items():
        main=item["macro5"]
        lines.append(f"| {metric} | {coeff(main['spearman'])} | {coeff(main['kendall'])} | {main['top3_matches']}/3 | {main['top1_regret_pp']:.2f} pp |")
    paired_predictor=v4/"prospective_unseen/PROSPECTIVE_UNSEEN11_PAIRED_PREDICTOR_DIAGNOSTIC.json"
    evidence["T4 paired predictor checks"]=paired_predictor
    difference=read(paired_predictor)["primary"]["CFIT_P4_Intrinsic_OIII_Ratio_Mpres minus v3_P4_unlabeled_pres_only"]
    lift_low,lift_high=difference["ci95"]
    lines.extend(["",f"P4 相对单用保真的 Spearman 差值为 {difference['difference']:+.3f}，"
                     f"配对来源重采样区间 [{lift_low:+.3f}, {lift_high:+.3f}]。"
                     "这是训练后诊断，继续沿用原额定预测与开发选择规则。"])
    lines.extend(["","## T5　混合、去混淆与容量","",
                  "零 ARC 赛道对 ARC-Challenge 与 ARC-Easy 做归一化题干与 8-gram 排除。"
                  "同源 MedMCQA 配对保持 ARC/CSQA/MedMCQA 的混合比例一致，改变第三源的保留样本。"
                  "BLUE 训练使用已探测题目的 Regime III 权重。分母的混淆与噪声来自锁定代理估计。",
                  "", "| 训练臂 | 独立开发 Macro5 | 平均变化 |",
                  "| --- | ---: | ---: |"])
    for name in sorted((v4/"exp_t5_mixing").glob("T5_T5_*.json")):
        data=read(name)
        delta=sum(x["delta_protected_pp"] for x in data["seeds"].values())/2
        lines.append(f"| {data['arm']} | {data['mean_protected_macro5']*100:.2f}% | {delta:+.2f} pp |")
        evidence[data["arm"]]=name
    cap=v["T5 capacity"]
    lines.extend(["",f"三目标共同容量的核范数下界为 {cap['lower_nuclear']:.6g}，"
                  f"全局上界为 {cap['upper_nuclear']:.6g}，数值间隙为 {cap['dual_gap']:.3g}。"
                  f"Rank-2 可行下界为 {cap['rank_feasible_lower']['2']['objective']:.6g}。"
                  f"是否触发预先定义的扩容条件为 {cap['expansion_triggered']}。","",
                  "## 官方 K4 封存评测","",
                  "八项历史对齐基准共 43,769 题，扩展后共 57,379 题。"
                  "同题配对按题重采样 10,000 次，SQuAD 按文章重采样。二值端点附精确 McNemar 检验。",
                  " [全量表](official/FINAL_8PLUS4_TABLE.md) · [逐题配对及生成指标](official/PAIRED_SIGNIFICANCE.json)","",
                  "| 控制器 | 来源/种子 | 八项等权 | 十二项等权 |",
                  "| --- | --- | ---: | ---: |"])
    for alias,body in selection["candidates"].items():
        legacy=table["macro"]["legacy_eight"][alias]
        full=table["macro"]["official_twelve"][alias]
        lines.append(f"| {alias} | {body['selected_arm']} | {legacy*100:.2f}% | {full*100:.2f}% |")
    lines.extend(["","## 证据文件 SHA-256","","| 文件 | SHA-256 |","| --- | --- |"])
    for name,file in evidence.items():
        lines.append(f"| {name} | {sha(file)} |")
    target=official/"V4_FINAL_REPORT_ZH.md"
    put(target,"\n".join(lines)+"\n")
    print("V4_REPORT_FINISHED",sha(target),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--official",type=Path,required=True)
    args=p.parse_args()
    report(args.work,args.official)
