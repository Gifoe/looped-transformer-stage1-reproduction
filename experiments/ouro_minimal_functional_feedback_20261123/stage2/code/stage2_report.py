"""Verify full few-shot cohorts and report paired fixed-budget comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark import holm, paired
from minimal import dump, load_jsonl, sha
from stage2_score import APPLE_BENCHMARKS


ARMS = ("native_k1", "native_k4", "loopcd_fixed_k4", "loopcd_adaptive_k4", "minimal_k4")
BASELINES = ARMS[1:-1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dataset(root, name):
    folder = root / "scores" / name
    lock = read(folder / "LOCK.json")
    manifest = lock["harness_manifest"]
    source = folder / "RESULTS.jsonl"
    rows = load_jsonl(source)
    n = manifest["totals"]["documents"]
    assert manifest["limit_per_task"] is None and lock["depths"] == [4]
    assert len(rows) == len({r["id"] for r in rows}) == n
    assert manifest["totals"]["requests"] == sum(len(x["choices"]) for x in rows)
    assert all(row["benchmark"] == name and set(row["scores"]) == set(ARMS) for row in rows)
    receipts = [read(p) for p in sorted(folder.glob("shard_*.cost.json"))]
    assert [x for r in receipts for x in r["ids"]] == [row["id"] for row in rows]
    metrics = set(manifest["tasks"][0]["metrics"]) & {"acc", "acc_norm"}
    gold = np.array([r["gold"] for r in rows])
    measures = {}
    for metric in sorted(metrics):
        masks = {}
        for arm in ARMS:
            preds = []
            for row in rows:
                score = np.asarray(row["scores"][arm])
                if metric == "acc_norm":
                    score = score / row["denominators"]
                preds.append(np.argmax(score))
            masks[arm] = np.asarray(preds) == gold
        measures[metric] = dict(correct={arm: int(mask.sum()) for arm, mask in masks.items()},
                                paired={old: paired(masks[old], masks["minimal_k4"]) for old in BASELINES})
    cost = dict(
        elapsed_shards_seconds=sum(x["wall_seconds"] for x in receipts),
        dense_loop_tokens=sum(x["dense_loop_tokens_padded"] for x in receipts),
        peak_allocated_bytes=max(x["peak_allocated_bytes"] for x in receipts),
        batches=sum(len(x["batches"]) for x in receipts),
        oom_retries=sum(x["oom_retries"] for x in receipts),
    )
    return dict(n=n, sha256=sha(source), fewshot=APPLE_BENCHMARKS[name], measurements=measures, cost=cost)


def run(root):
    summary = {key: dataset(root, key) for key in APPLE_BENCHMARKS}
    for base in BASELINES:
        holm([m["paired"][base] for d in summary.values() for m in d["measurements"].values()])
    dump(root / "COMPARISON.json", summary)
    lines = [
        "# 七项 few-shot 基准 · Ouro-1.4B Base K4",
        "",
        "本阶段复用已锁定的 rank-8 线性反馈。示例数依据论文表 A2。"
        "逐题请求由 lm-eval v0.4.13 导出，固定 seed 1234 和数据修订。"
        "论文未公开该表使用的 Harness 版本、示例随机种子和全部任务的报告列。"
        "这里在同一请求上重新评估原生、固定和自适应 Logit LoopCD、以及最简反馈。"
        "K1 提供原生参考分数。"
        " [原论文表 A2](https://arxiv.org/html/2610.02185v1#A2.T2)"
        " [Harness 任务定义](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.13/lm_eval/tasks)",
        "",
        "| 评测 | Shots | 题数 | 指标 | Native K1 | Native K4 | 固定 LoopCD K4 | 自适应 LoopCD K4 | 最简反馈 K4 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for key, item in summary.items():
        n = item["n"]
        for metric, m in item["measurements"].items():
            values = [m["correct"][a] for a in ARMS]
            cells = " | ".join(f"{100*c/n:.2f}% ({c})" for c in values)
            lines.append(f"| {key} | {item['fewshot']} | {n} | {metric} | {cells} |")
    lines += [
        "",
        "| 评测 | 主报告指标 | 最简对自适应救回/退化 | 净差 pp [95% CI] | Holm p | 联合评测秒 | 峰值 GiB |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for key, item in summary.items():
        metric = "acc_norm" if "acc_norm" in item["measurements"] else "acc"
        p = item["measurements"][metric]["paired"]["loopcd_adaptive_k4"]
        lo, hi = p["ci95_pp"]
        cost = item["cost"]
        lines.append(
            f"| {key} | {metric} | {p['rescued']}/{p['damaged']} | {p['delta_pp']:+.2f} "
            f"[{lo:+.2f}, {hi:+.2f}] | {p['p_holm']:.3g} | {cost['elapsed_shards_seconds']:.1f} "
            f"| {cost['peak_allocated_bytes']/2**30:.2f} |"
        )
    lines += [
        "",
        "联合评测时间包含所有对照。原生前缀在离线评测中复用，基于冻结状态分支的成本不能当作最简方法的独立部署延迟。",
        "MMLU 和 WinoGrande 使用任务定义的 acc。其余双指标任务两列均保留，主表逐题权重相同。",
        "救回和退化均按同一题目配对。Holm 校正对同一对照的全部任务与指标端点执行。",
    ]
    (root / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
