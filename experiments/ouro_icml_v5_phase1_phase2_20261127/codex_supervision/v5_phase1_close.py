"""Close Phase 1 after the strict CSBP failure and complete PEFT official confirmation."""
import argparse
from pathlib import Path
from statistics import mean

from v4_common import save_json, sha256_file
from v5_matched_peft import METHODS, SEEDS, SOURCES, STATIC, read
from v5_peft_official import pin_source
from run_full import LEGACY

PRIMARY={"arc_challenge":"acc_norm","arc_easy":"acc_norm","sciq":"acc_norm","mmlu":"acc",
         "hellaswag":"acc_norm","winogrande":"acc","piqa":"acc_norm","mmlu_pro":"correct"}


def paired_s1_holdout(folder):
    """Compare shared held-out teacher agreement without counting the same item twice."""
    from aggregate_v3 import paired
    import numpy as np
    reference=[pin_source(folder,"S1","M0_InLoop_Rank1",seed)[2]["selection"]["flags"] for seed in SEEDS]
    if any(len(values)!=153 for values in reference):
        raise ValueError("The S1 paired population is incomplete")
    reference=np.asarray(reference,dtype=float)
    details={}
    for method in METHODS[1:]+STATIC:
        challenger=np.asarray([pin_source(folder,"S1",method,seed)[2]["selection"]["flags"] for seed in SEEDS],dtype=float)
        if challenger.shape!=reference.shape:
            raise ValueError("The paired S1 predictions differ in length")
        details[method]={"seed_average_paired":paired(challenger.mean(0),reference.mean(0),None),
                         "per_seed":{str(seed):paired(challenger[j],reference[j],None) for j,seed in enumerate(SEEDS)}}
    return details


def main(root):
    a=root/"track_a_csbp/preflight"
    verdict_path=a/"CSBP_PREFLIGHT_VERDICT.json"
    verdict=read(verdict_path)
    if len(verdict["matrix"])!=24 or verdict["preflight_lock_sha256"]!=sha256_file(a/"PREFLIGHT_LOCK.json"):
        raise ValueError("Missing the locked strict diagnostic")
    if verdict["status"]!="FAIL":
        raise ValueError("Wait for strict 1B and conditional symmetry before 1D")
    b=root/"track_b_baselines"
    numbers={}
    for method in METHODS+STATIC:
        each={}
        for source in SOURCES:
            arms=[pin_source(b,source,method,seed)[2] for seed in SEEDS]
            each[source]=dict(
                params=arms[0]["training"]["n_params"],n_seeds=len(arms),
                macro5=mean(x["development"]["macro5"] for x in arms),
                non_mmlu4=mean(x["development"]["non_mmlu4"] for x in arms),
                cloze3=mean(x["development"]["cloze3"] for x in arms),
                mmlu_dev=mean(x["development"]["mmlu_dev158"]["acc"] for x in arms),
                mmlu_pro=mean(x["development"]["mmlu_pro_val"]["correct"] for x in arms),
                delta_nll=mean(x["unlabeled"]["delta_nll"] for x in arms),
                eos_shift=mean(x["unlabeled"]["eos_abs_shift"] for x in arms),
                latency_ms_per_question=mean(x["latency_ms_per_question"] for x in arms),
                train_seconds=mean(x["training"].get("train_seconds",x["training"].get("fit_seconds",0.)) for x in arms),
                peak_mib=max(x["training"].get("peak_allocated_mib",0.) for x in arms),
                weights=[x["weights_sha256"] for x in arms],
            )
        numbers[method]=each
    heldout=paired_s1_holdout(b)
    official_path=b/"PEFT_FULL8_PAIRED.json"
    official=read(official_path)
    selection=root/"SELECTION_PHASE1_TRACK_B.json"
    if official["n"]!=43769 or official["selection_sha256"]!=sha256_file(selection) or any(
            set(res)!=set(LEGACY) for res in official["methods"].values()):
        raise ValueError("Track B official outputs are incomplete")
    finalists={}
    for method,tests in official["methods"].items():
        scores={}
        for task, item in tests.items():
            metric=PRIMARY[task]
            paired=item["paired"]["best_zerolabel_v4"][metric]
            scores[task]=dict(native=item["paired"]["native"][metric]["baseline"],anchor=paired["baseline"],
                              accuracy=paired["new"],paired=paired,cost=item["costs"]["new"],generation=item["generation"]["new"])
        finalists[method]=dict(task=scores,macro=mean(x["accuracy"] for x in scores.values()),
                               baseline_macro=mean(x["anchor"] for x in scores.values()))
    summary=dict(
        statement="PHASE2_HOLD",phase1_complete_at_prespecified_gate=True,
        gate1=dict(status="FAIL",reason="Strict 1A did not yield an eligible diagnostic recipe.",
                   preflight_sha256=sha256_file(verdict_path),closest=verdict["nearest"]),
        gate2=dict(status="NOT_ESTABLISHED_FOR_PURE_PROSE",reason="No eligible CSBP checkpoint or symmetric cohort."),
        gate3=dict(status="SIGNED_LOCAL_AUDIT",corpus_audit_sha256=sha256_file(a/"CORPUS_DEDUP_AUDIT.json"),
                   selection_sha256=sha256_file(selection),official_paired_sha256=sha256_file(official_path)),
        available_1A=verdict["matrix"],available_1C=numbers,official=finalists,
        exploratory_matched_s1_teacher_agreement=heldout,
        old_T2B=dict(prose_rank_lam0=dict(delta_macro5_pp=3.60,delta_nll=2.221),
                     prose_rank_lam3=dict(delta_macro5_pp=2.40,delta_nll=.014)),
        conditional_1B="Blocked by 1A hard gate",conditional_symmetry="No eligible CSBP checkpoint",
    )
    dest=root/"PHASE1_MASTER_SUMMARY.json"
    save_json(dest,summary)
    save_json(b/"PEFT_STEERING_BENCHMARK.json",dict(matrix=numbers,official=finalists))
    table=["# Phase 1 严格准入报告","", "**PHASE2_HOLD**","",
           "Track A 在附件预设 1A 门控处结束。原句恢复仅用于离线诊断。","",
           "| 1A 近门槛组合 | 四选项样本 | 过滤后恢复率 | 读出差（pp） | 预计可用探针 |",
           "|---|---:|---:|---:|---:|"]
    for name in verdict["nearest"]:
        x=verdict["matrix"][name]
        table.append(f"| {name} | {x['n']} | {x.get('filtered_mc',0):.2%} | {x.get('filtered_gap_pp',0):.2f} | {x['projected_filtered']} |")
    btable=["| Track B 方法 | 参数量 | S1 Macro5 | S2 Macro5 | S3 Macro5 | S1 每题延迟（ms） |",
            "|---|---:|---:|---:|---:|---:|"]
    for method,x in numbers.items():
        btable.append(f"| {method} | {x['S1']['params']:,} | {x['S1']['macro5']:.2%} | {x['S2']['macro5']:.2%} | {x['S3']['macro5']:.2%} | {x['S1']['latency_ms_per_question']:.1f} |")
    btable.extend(["","| K4 全量官方八项 | 题数 | V4 零人类标签锚点 | "+" | ".join(finalists)+" |",
                   "|---|---:|---:|"+"---:|"*len(finalists)])
    left=next(iter(finalists.values()))
    for task,n in LEGACY.items():
        values=" | ".join(f"{x['task'][task]['accuracy']:.2%}" for x in finalists.values())
        btable.append(f"| {task} | {n:,} | {left['task'][task]['anchor']:.2%} | {values} |")
    btable.append(f"| 等权平均 | 43,769 | {left['baseline_macro']:.2%} | "+" | ".join(f"{x['macro']:.2%}" for x in finalists.values())+" |")
    table.extend(["","另见主 JSON 中 153 题 S1 教师一致率的配对诊断。该留出折也用于静态剂量和基线定榜，相关区间用于描述观测结果。","",*btable,"","逐题配对 McNemar、10,000 次 bootstrap、生成合法率和实际成本见 track_b_baselines/PEFT_FULL8_PAIRED.json。",""])
    for path,lines in [(root/"PHASE1_ICML_GATE_REPORT.md",table),(b/"PEFT_STEERING_SUMMARY.md",btable)]:
        tmp=path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines)+"\n",encoding="utf-8")
        tmp.replace(path)
    print("TRACK_B_BASELINES_DONE PHASE1_ALL_COMPLETE PHASE2_HOLD",sha256_file(dest),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root)

