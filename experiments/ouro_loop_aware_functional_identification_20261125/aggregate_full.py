"""Pair the final checkpoint with the seven historical controllers."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

import evaluate_core as core
from evaluate_benchmarks import candidate_row, mmlu_metrics
import minimal
from run_full import EXTRA, LEGACY, read
from task_data import load


BASELINES = ("native", "fixed", "adaptive", "rank8", "t1", "t2", "mixed")


def closed(path, expected_ids, freeze, weight=None, protocol=None):
    receipt = read(path.with_suffix(".json"))
    spec = receipt["plan"]
    if spec["depth"] != 4 or spec["freeze_sha256"] != minimal.sha(freeze):
        raise ValueError(f"Unmatched K4 protocol: {path}")
    if weight is not None and spec["checkpoint_sha256"] != weight:
        raise ValueError(f"Unmatched checkpoint: {path}")
    if protocol is not None:
        for name in ("input_sha256", "config", "batch_size", "capacity_tokens", "max_new"):
            if spec.get(name) != protocol.get(name):
                raise ValueError(f"Unmatched {name}: {path}")
    rows, cost = core.verified(path, spec, expected_ids)
    return rows, cost, receipt


def legacy_rows(a, task, count):
    import seven_unified as scorer
    import stage2_score as stage

    stage.APPLE_BENCHMARKS = dict.fromkeys(stage.APPLE_BENCHMARKS, 0)
    scorer.TASKS = {name: (0, n, metrics) for name, (_, n, metrics) in scorer.TASKS.items()}
    generation = task == "mmlu_pro"
    mode = "generated" if generation else "candidate"
    target = a.out / "legacy" / f"{task}__new__k4__{mode}.jsonl"
    receipt = read(target.with_suffix(".json"))
    runtime = receipt["runtime"]
    if generation:
        import generation_followup_run as decoder
        from transformers import AutoTokenizer
        import unified_content as history

        source = a.followup / "data/mmlu_pro.parquet"
        archived, old_runtime, _ = history.read_history(
            a, task, read(source.with_suffix(".json")), history.PROTOCOL[task]
        )
        if runtime["files_sha256"] != old_runtime["files_sha256"] or runtime["model_sha256"] != old_runtime["model_sha256"]:
            raise ValueError("MMLU-Pro historical model/tokenizer differs")
        tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, local_files_only=True)
        items = [item for batch in decoder.batches(
            decoder.prompt_items(source, task, tokenizer), 8, 96, 14500
        ) for item in batch]
        ids = [item.key for item in items]
        refs = {}
        for name, key in zip(("native", "fixed", "adaptive", "rank8"), history.BASELINES, strict=True):
            raw = archived[key]
            refs[name] = (mmlu_metrics(raw["rows"], items, decoder), raw["cost"], {"rows_sha256": raw["sha256"]})
    else:
        index = scorer.json_read(a.stage2 / "COMPARISON.json")
        panel = scorer.check_archive(a, stage, runtime, task, index)
        ids = [r["id"] for r in panel.requests]
        keys = ("native_k4", "loopcd_fixed_k4", "loopcd_adaptive_k4", "minimal_k4")
        # The archive timed a joint run. Standalone method time is unavailable.
        cost = {"archived_joint_scoring_cost": panel.public["archived_joint_scoring_cost"]}
        doc = {"rows_sha256": panel.public["historic_result_uncompressed_sha256"]}
        refs = {name: ([candidate_row(r, key) for r in panel.references], cost, doc)
                for name, key in zip(("native", "fixed", "adaptive", "rank8"), keys, strict=True)}
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError(f"Incomplete official population {task}")
    official = a.out / "FREEZE.json"
    own = closed(target, ids, official, read(official)["arms"]["new"]["sha256"])
    for name in ("mixed", "t1", "t2"):
        root = a.release / f"{name}_full/results"
        freeze = root / "FREEZE.json"
        slot = read(freeze)["arms"][name]
        refs[name] = closed(root / "predictions" / f"{task}__{name}__k4__{mode}.jsonl",
                            ids, freeze, slot["sha256"], own[2]["plan"])
    for name, (rows, _, _) in refs.items():
        if [r["gold"] for r in rows] != [r["gold"] for r in own[0]]:
            raise ValueError(f"Historical gold mismatch {name}/{task}")
    return {"new": own, **refs}, None


def extra_rows(a, task, count, mode):
    panel = load(a.panel / f"test_{task}.jsonl")
    ids = [r["sample_id"] for r in panel]
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError(f"Incomplete official population {task}")
    freeze = a.out / "FREEZE.json"
    arms = read(freeze)["arms"]
    results = {}
    for method in ("new", *BASELINES):
        path = a.out / "extra" / f"{task}__{method}__k4__{mode}.jsonl"
        weight = None if mode == "standard" else arms[method]["sha256"]
        results[method] = closed(path, ids, freeze, weight)
    clusters = [r["title"] for r in panel] if task == "squad" else None
    return results, clusters


def pair(a, b, clusters, seed=261125137):
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    delta = right - left
    rng = np.random.default_rng(seed)
    n = len(delta)
    binary = set(a + b) <= {0, 1}
    advances, losses = int((delta > 0).sum()), int((delta < 0).sum())
    if clusters is None:
        values, counts = np.unique(delta, return_counts=True)
        boot = rng.multinomial(n, counts / n, size=5000) @ values / n
    else:
        _, ids = np.unique(clusters, return_inverse=True)
        counts = np.bincount(ids)
        totals = np.bincount(ids, weights=delta)
        draws = rng.multinomial(len(counts), np.full(len(counts), 1 / len(counts)), size=5000)
        boot = (draws @ totals) / (draws @ counts)
    p = float(binomtest(advances, advances + losses).pvalue) if binary and (advances + losses) else (1. if binary else None)
    return dict(n=n, baseline=float(left.mean()), new=float(right.mean()), change=float(delta.mean()),
                improved=advances, degraded=losses, ci95=np.quantile(boot, [.025, .975]).tolist(),
                exact_mcnemar=p, cluster_unit="article" if clusters is not None else "question")


def diagnostics(rows):
    tok = [r.get("generated_tokens", r.get("tokens")) for r in rows]
    if tok[0] is None:
        return {}
    length = [len(x) for x in tok]
    return {"mean_new_tokens": float(np.mean(length)), "median_new_tokens": float(np.median(length)),
            "eos_before_two_tokens": sum(r.get("stop") == "eos" and n < 2 for r, n in zip(rows, length, strict=True)),
            "token_limit": sum(r.get("stop") == "token_limit" for r in rows)}


def metrics(rows):
    keys = {"acc", "acc_norm", "correct", "valid", "parseable", "strict", "strict_format",
            "standard_exact", "standard_f1", "f1", "premature_eos", "eos_before_16", "repetitive"}
    return keys.intersection(rows[0]["metrics"])


def details(task, mode, results, clusters):
    own = results["new"][0]
    paired = {}
    for name in BASELINES:
        rows = results[name][0]
        paired[name] = {key: pair([r["metrics"][key] for r in rows], [r["metrics"][key] for r in own], clusters)
                        for key in sorted(metrics(own).intersection(metrics(rows)))}
    return dict(task=task, mode=mode, paired=paired, costs={k: v[1] for k, v in results.items()},
                diagnostics={k: diagnostics(v[0]) for k, v in results.items()},
                prediction_sha256={k: v[2]["rows_sha256"] for k, v in results.items()})


def main(a):
    sys.path[:0] = [str(a.unified), str(a.followup), str(a.stage2 / "code")]
    lock = read(a.out / "FREEZE.json")
    rows, task_primary, flat = [], [], []
    for task, count in (LEGACY | EXTRA).items():
        if task in LEGACY:
            modes = ("generated",) if task == "mmlu_pro" else ("candidate",)
            primary_mode = modes[0]
        else:
            modes = ("candidate", "content", "letter") if task in ("csqa", "obqa") else ("standard",)
            primary_mode = modes[0]
        for mode in modes:
            data = legacy_rows(a, task, count) if task in LEGACY else extra_rows(a, task, count, mode)
            row = details(task, mode, *data)
            rows.append(row)
            target = lock["primary"][task]
            if mode == primary_mode:
                task_primary.append((task, target, count, row))
            for baseline, comparisons in row["paired"].items():
                for metric, comparison in comparisons.items():
                    flat.append({"benchmark": task, "mode": mode, "metric": metric, "baseline": baseline, **comparison})
    minimal.dump(a.out / "RESULTS.json", {"freeze_sha256": minimal.sha(a.out / "FREEZE.json"),
                                        "primary": lock["primary"], "tasks": rows})
    with (a.out / "RESULTS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    header = ["Benchmark", "N", "Metric", *BASELINES, "New"]
    table = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for task, metric, n, row in task_primary:
        p = row["paired"]
        values = [p[name][metric]["baseline"] for name in BASELINES] + [p["native"][metric]["new"]]
        table.append("| " + " | ".join([task, f"{n:,}", metric] + [f"{100*v:.2f}%" for v in values]) + " |")
    for label, jobs in (("历史八项等权平均", task_primary[:8]), ("十二项等权平均", task_primary)):
        scores = [[row["paired"][k][metric]["baseline"] for _, metric, _, row in jobs] for k in BASELINES]
        scores.append([row["paired"]["native"][metric]["new"] for _, metric, _, row in jobs])
        table.append("| " + " | ".join([label, "", ""] + [f"{100*np.mean(x):.2f}%" for x in scores]) + " |")
    page = ["# 完整官方 K4 性能", "", *table, "", "[逐项配对和置信区间](RESULTS.csv) · [计算量与格式](RESULTS.json)", ""]
    (a.out / "RESULTS.md").write_text("\n".join(page), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "release", "panel", "stage2", "minimal", "model", "followup", "unified", "historical"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    main(parser.parse_args())
