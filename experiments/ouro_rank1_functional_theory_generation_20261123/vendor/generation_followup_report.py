"""Audit matched K4 greedy results, preserving full test cohorts and paired outcomes."""
import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from benchmark import holm, paired
from generation_followup_run import ARMS, TASKS
from minimal import dump, load_jsonl, sha

HARNESS_FLEXIBLE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
HARNESS_STOP = re.compile(r"Question:|</s>|<\|im_end\|>")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def number(value):
    if value is None:
        return None
    try:
        return Fraction(value.replace(",", "").replace("$", "").strip())
    except (ValueError, ZeroDivisionError):
        return None


def harness_normalize(value):
    return re.sub(r"\.$", "", value.replace(",", "").replace("$", "").lower())


def harness_match(text, gold):
    before_stop = HARNESS_STOP.split(text, maxsplit=1)[0]
    hits = HARNESS_FLEXIBLE.findall(before_stop)
    if not hits:
        return False
    pred = next(item.strip() for item in hits[-1] if item)
    return harness_normalize(pred) == harness_normalize(gold)


def detail(result, dataset):
    base = result / dataset
    frozen = read(base / "LOCK.json")
    name = {
        "gsm8k": "generation_followup_run_gsm8k.py",
        "mmlu_pro": "generation_followup_run_reference.py",
        "humaneval": "generation_followup_run_reference.py",
        "mbpp": "generation_followup_run.py",
    }[dataset]
    assert sha(Path(__file__).with_name(name)) == frozen["code_sha256"]
    n = frozen["dataset"]["rows"]
    predictions, evaluations, timing = {}, {}, {}
    for arm in ARMS:
        path = base / f"{arm}.jsonl"
        rows = load_jsonl(path)
        assert len(rows) == len({r["id"] for r in rows}) == n
        predictions[arm] = {r["id"]: r for r in rows}
        costs = [read(q) for q in sorted((base / arm).glob("*.cost.json"))]
        assert [p for x in costs for p in x["ids"]] == [r["id"] for r in rows]
        timing[arm] = dict(
            seconds=sum(c["seconds"] for c in costs),
            median_batch_seconds=float(np.median([c["seconds"] for c in costs])),
            peak_allocated_bytes=max(o["peak_bytes"] for o in costs),
            mean_new_tokens=sum(c["new_tokens"] for c in costs) / n,
            limit_hits=sum(r["stop"] == "limit" for r in rows),
            stop_counts=dict(Counter(r["stop"] for r in rows)),
            sha256=sha(path),
        )
        if dataset in {"humaneval", "mbpp"}:
            grades = base / f"{arm}.evalplus.jsonl"
            records = load_jsonl(grades)
            evaluations[arm] = {r["id"]: r for r in records}
            assert len(records) == n and set(evaluations[arm]) == predictions[arm].keys()
            assert all(r["source_sha256"] == hashlib.sha256(
                predictions[arm][r["id"]]["generated"].encode()).hexdigest() for r in records)
            timing[arm]["grading_sha256"] = sha(grades)
            timing[arm]["has_non_docstring_function_body"] = sum(has_function_body(x["sanitized"]) for x in records)
        else:
            timing[arm]["valid"] = sum(r["predicted"] is not None for r in rows)
    keys = list(predictions["native"])
    assert all(set(p) == set(keys) for p in predictions.values())
    assert all(all(records[k]["gold"] == predictions["native"][k]["gold"] for k in keys)
               for records in predictions.values())
    masks = {}
    for arm, records in predictions.items():
        if dataset in {"humaneval", "mbpp"}:
            masks[arm] = {(f"{prefix}_pass_at_1"): np.array([
                evaluations[arm][i][prefix] == "pass" for i in keys
            ]) for prefix in ("base", "plus")}
        elif dataset == "gsm8k":
            masks[arm] = {"strict_match": np.array([
                number(records[k]["predicted"]) is not None
                and number(records[k]["predicted"]) == number(records[k]["gold"]) for k in keys
            ]), "harness_flexible_extract": np.array([
                harness_match(records[k]["generated"], records[k]["gold"]) for k in keys
            ])}
        else:
            masks[arm] = {"greedy_option_accuracy": np.array([
                records[k]["predicted"] == records[k]["gold"] for k in keys
            ])}
    measures = {}
    for metric in masks["feedback"]:
        correct = {arm: int(masks[arm][metric].sum()) for arm in ARMS}
        p = {other: paired(masks[other][metric], masks["feedback"][metric]) for other in ARMS[:-1]}
        measures[metric] = dict(correct=correct, paired_vs_baselines=p)
    out = dict(n=n, metrics=measures, timing=timing, frozen=frozen,
               cache_parity=read(base / "CACHE_PARITY.json"))
    if dataset == "mmlu_pro":
        baseline = max(ARMS[:-1], key=measures["greedy_option_accuracy"]["correct"].get)
        before, after = (masks[k]["greedy_option_accuracy"] for k in (baseline, "feedback"))
        va, vb = (np.array([predictions[k][i]["predicted"] is not None for i in keys])
                  for k in (baseline, "feedback"))
        common = va & vb
        out["format_breakdown"] = dict(
            baseline=baseline, baseline_valid=int(va.sum()), feedback_valid=int(vb.sum()),
            baseline_invalid_to_feedback_correct=int((~va & after).sum()),
            baseline_correct_to_feedback_invalid=int((before & ~vb).sum()),
            both_valid=int(common.sum()), both_valid_baseline_correct=int(before[common].sum()),
            both_valid_feedback_correct=int(after[common].sum()),
            both_valid_paired=paired(before[common], after[common]),
        )
    return out


def has_function_body(code):
    try:
        nodes = ast.parse(code).body
    except SyntaxError:
        return False
    return any(
        isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and
        any(not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Constant) for stmt in fn.body)
        for fn in nodes
    )


def run(result, out, partial, ledger):
    summary = {}
    for key in ("gsm8k", "mmlu_pro", "humaneval", "mbpp"):
        result_files = [result / key / f"{arm}.jsonl" for arm in ARMS]
        if key in {"humaneval", "mbpp"}:
            result_files.extend(result / key / f"{arm}.evalplus.jsonl" for arm in ARMS)
        if all(p.exists() for p in result_files):
            summary[key] = detail(result, key)
        elif not partial:
            raise RuntimeError(f"Full benchmark missing: {key}")
    if not summary:
        raise RuntimeError("No completed generation benchmark")
    for base in ARMS[:-1]:
        holm([p["paired_vs_baselines"][base] for d in summary.values() for p in d["metrics"].values()])
    lines = [
        "# Ouro-1.4B Base 生成评测",
        "",
        "共享 rank 8 反馈沿用冻结的 32,768 参数权重。四组对照均以 K4、同题面、0 个示例和完整词表贪心解码运行。"
        "LoopCD 的固定系数为 0.3，自适应上限为 0.5，两者用 K1 作参考。"
        "论文的 GSM8K/MMLU-Pro 使用示例，这里的 zero-shot 结果是独立评测。"
        " [LoopCD 附录协议](https://arxiv.org/html/2610.02185v1#A2.T3)",
        "",
        "MMLU-Pro 用完整词表，接受输出开头的合法选项字母。GSM8K 严格指标从 #### 行提取数值。"
        "补充指标使用 Harness 0.4.13 的停止字符串、末次数字提取和字符串归一化，复核同一批保存的输出。"
        " [GSM8K 评分规则](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/tasks/gsm8k/gsm8k.yaml)",
        "代码题由 EvalPlus 0.3.1 的完整原版和扩展测试评分。HumanEval 使用原始完成型题面。"
        "MBPP 使用问题说明和题目提供的公开断言，以明确函数接口。",
        "",
        "| 任务 | 全量题数 | 指标 | Native K4 | 固定 LoopCD | 自适应 LoopCD | 共享反馈 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for tag, x in summary.items():
        n = x["n"]
        for metric, values in x["metrics"].items():
            cells = " | ".join(f"{100 * values['correct'][arm]/n:.2f}% ({values['correct'][arm]})" for arm in ARMS)
            lines.append(f"| {tag} | {n} | {metric} | {cells} |")
    lines += [
        "",
        "| 任务 | 指标 | 同任务最强基线 | 救回/退化 | 净差 pp [95% CI] | Holm p |",
        "|---|---|---|---:|---|---:|",
    ]
    for tag, x in summary.items():
        for metric, row in x["metrics"].items():
            base = max(ARMS[:-1], key=row["correct"].get)
            p = row["paired_vs_baselines"][base]
            a, b = p["ci95_pp"]
            lines.append(f"| {tag} | {metric} | {base} | {p['rescued']}/{p['damaged']} | "
                         f"{p['delta_pp']:+.2f} [{a:+.2f},{b:+.2f}] | {p['p_holm']:.3g} |")
    if "mmlu_pro" in summary:
        d = summary["mmlu_pro"]["format_breakdown"]
        lines += [
            "",
            f"MMLU-Pro 最强基线有 {d['baseline_valid']:,} 个合法字母，反馈有 {d['feedback_valid']:,} 个。"
            f"基线无效而反馈答对的有 {d['baseline_invalid_to_feedback_correct']:,} 题，"
            f"反向的有 {d['baseline_correct_to_feedback_invalid']:,} 题。"
            f"两个方案都输出合法字母的 {d['both_valid']:,} 题中，"
            f"基线和反馈分别答对 {d['both_valid_baseline_correct']:,} 和 {d['both_valid_feedback_correct']:,} 题。"
            "该子集按两个模型的输出定义，作为描述性分解。",
        ]
    if "humaneval" in summary:
        d = summary["humaneval"]["timing"]
        lines += ["", "HumanEval 的清洗结果中，包含非纯文档字符串函数体的条数依次为 "
                  + " / ".join(str(d[a]["has_non_docstring_function_body"]) for a in ARMS)
                  + "（Native、固定、自适应、反馈）。该项用于定位生成失败，有效性以执行测试为准。"]
    lines += [
        "",
        "| 任务 | 各组总秒数，Native/固定/自适应/反馈 | 反馈平均生成 token | 反馈达到 token 上限 | 反馈峰值 GiB |",
        "|---|---|---:|---:|---:|",
    ]
    for task, x in summary.items():
        t = x["timing"]
        total = " / ".join(f"{t[arm]['seconds']:.1f}" for arm in ARMS)
        f = t["feedback"]
        lines.append(f"| {task} | {total} | {f['mean_new_tokens']:.2f} | {f['limit_hits']} | "
                     f"{f['peak_allocated_bytes']/2**30:.2f} |")
    use = 0
    total = 0
    for line in ledger.read_text().splitlines():
        fields = line.split("\t")
        if fields[0] == "END":
            elapsed = int(fields[4])
            total += elapsed
            if fields[1].startswith("gen_followup_"):
                use += elapsed
    lines += ["", f"追加阶段的串行 GPU 台账为 {use} 秒。同一张卡的整条研究台账累计为 {total} 秒。",
              "共享权重、数据版本、配对区间、实际生成长度和格式有效数均保存在 COMPARISON.json。"]
    if pending := sorted(TASKS.keys() - summary.keys()):
        lines += ["", "当前已有全量生成评分的任务见上表。其它任务状态为 " + ", ".join(pending) + "。"]
    dump(out / "COMPARISON.json", dict(benchmarks=summary, cumulative_gpu_seconds=total, generation_gpu_seconds=use))
    (out / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--partial", action="store_true")
    a = p.parse_args()
    run(a.results, a.out, a.partial, a.ledger)
