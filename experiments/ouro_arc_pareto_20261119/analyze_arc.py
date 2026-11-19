"""Freeze development choices, and score released per-question outputs."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from run_arc import SEED, dump


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def vectors(rows, name, metric):
    winners, gaps, ties = [], [], []
    for row in rows:
        scores = np.array(row["scores"][name])
        if metric == "acc_norm":
            scores /= row["characters"]
        winners.append(scores.argmax())
        best_two = np.sort(scores)[-2:]
        gaps.append(best_two[1] - best_two[0])
        ties.append(best_two[1] == best_two[0])
    return np.array(winners), np.array(gaps), np.array(ties)


def paired(base, current, boot=True):
    improved = int((~base & current).sum())
    damaged = int((base & ~current).sum())
    delta = current.astype(int) - base.astype(int)
    result = {
        "wrong_to_correct": improved, "correct_to_wrong": damaged,
        "delta_pp": 100 * float(delta.mean()),
        "mcnemar_two_sided_p": float(binomtest(improved, improved + damaged).pvalue)
        if improved + damaged else 1.0,
    }
    if boot:
        proportions = np.bincount(delta + 1, minlength=3) / len(delta)
        draws = np.random.default_rng(SEED).multinomial(len(delta), proportions, size=20000)
        result["delta_bootstrap_95_pp"] = (100 * np.quantile(
            (draws[:, 2] - draws[:, 0]) / len(delta), [0.025, 0.975])).tolist()
    return result


def loops(name):
    return int(name[-1])


def summary(rows):
    methods = list(rows[0]["scores"])
    gold = np.array([row["gold"] for row in rows])
    n = len(rows)
    result = {}
    for metric in ("acc", "acc_norm"):
        table = {}
        outputs = {name: vectors(rows, name, metric) for name in methods}
        native = {name: pred == gold for name, (pred, _, _) in outputs.items() if name.startswith("native")}
        for name, (pred, _, ties) in outputs.items():
            good = pred == gold
            depth = loops(name) if "_k" in name else 2
            table[name] = {
                "correct": int(good.sum()), "n": n, "accuracy_pct": 100 * float(good.mean()),
                "ties": int(ties.sum()), "loop_budget": depth,
                "vs_same_depth_native": paired(native[f"native_k{depth}"], good),
            }
            if "native_k4" in native:
                table[name]["vs_native_k4"] = paired(native["native_k4"], good)
            if name in native and f"native_k{depth-1}" in native:
                table[name]["vs_previous_native"] = paired(native[f"native_k{depth-1}"], good)
        result[metric] = table
    return result


def gate_oracle(oracle):
    rows = oracle["acc_norm"]
    result = {}
    for component in ("q", "k", "v"):
        actual = rows[f"{component}_teacher"]
        controls = [rows[f"{component}_{which}"]["correct"] for which in ("random", "previous_position")]
        comparison = actual["vs_same_depth_native"]
        result[component] = dict(
            delta_vs_native_pp=comparison["delta_pp"],
            p=comparison["mcnemar_two_sided_p"],
            better_than_both_matched_controls=actual["correct"] > max(controls),
            passes_screen=(comparison["delta_pp"] >= 2 and comparison["mcnemar_two_sided_p"] < 0.05 / 3
                           and actual["correct"] > max(controls)),
        )
    return result


def freeze(out):
    validation = load(out / "validation_predictions.jsonl")
    oracle = summary(load(out / "oracle_predictions.jsonl"))
    dev = summary(validation)
    dump(out / "validation_metrics.json", dev)
    dump(out / "oracle_metrics.json", oracle)
    primary = dev["acc_norm"]
    baseline, challengers = {}, {}
    for depth in range(1, 5):
        comparisons = [name for name in primary if loops(name) == depth and name.startswith(("native", "loopcd"))]
        # Match metric and depth, and prefer one-head baselines in an accuracy tie.
        is_low_cost = lambda name: name.startswith(("native", "loopcd_hidden"))
        baseline[str(depth)] = max(comparisons, key=lambda n: (primary[n]["correct"], is_low_cost(n), n))
        if depth >= 3:
            name = f"secant_k{depth}"
            comparison = baseline[str(depth)]
            delta = paired(vectors(validation, comparison, "acc_norm")[0] == [r["gold"] for r in validation],
                           vectors(validation, name, "acc_norm")[0] == [r["gold"] for r in validation])
            challengers[str(depth)] = dict(method=name, baseline=comparison, **delta,
                                          passes_screen=delta["delta_pp"] >= 2 and delta["mcnemar_two_sided_p"] < 0.025)
    # Replay same question-level observed predictions. This explicitly includes
    # no claim of real acceleration or measured deployment latency.
    candidates = [row for row in validation]
    _, margins, _ = vectors(candidates, baseline["2"], "acc_norm")
    threshold = float(np.median(margins))
    dump(out / "freeze.json", {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "development_rows": len(validation),
        "primary": "acc_norm", "base_commit": "f525239970b70c7ce28fe9fe91d95859653096cc",
        "strong_baseline_by_depth": baseline,
        "secant_screen": challengers, "oracle_training_screen": gate_oracle(oracle),
        "adaptive_depth_replay": {"threshold": threshold, "shallow": baseline["2"], "deep": baseline["4"]},
        "config_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("run_arc.py", "diagnostics.py", "analyze_arc.py")},
        "validation_scores_sha256": hashlib.sha256((out / "validation_predictions.jsonl").read_bytes()).hexdigest(),
        "oracle_scores_sha256": hashlib.sha256((out / "oracle_predictions.jsonl").read_bytes()).hexdigest(),
    })


def analyze(out):
    config = json.loads((out / "freeze.json").read_text())
    rows = load(out / "test_predictions.jsonl")
    assert len(rows) == len({r["id"] for r in rows}) == 1172
    data = summary(rows)
    gold = np.array([row["gold"] for row in rows])
    detail_path = out / "test_verdicts.csv"
    route = config["adaptive_depth_replay"]
    _, margin, _ = vectors(rows, route["shallow"], "acc_norm")
    shallow = margin >= route["threshold"]
    depths = np.where(shallow, 2, 4)
    work = np.array([sum(row["scored_input_tokens"]) for row in rows])
    heads = np.array([sum(row["answer_tokens"]) for row in rows])
    runtime = json.loads((out / "runtime.json").read_text())
    latency = json.loads((out / "latency.json").read_text())
    verdicts, routing = {}, {}
    h = runtime["hidden_size"]
    v = runtime["vocabulary_size"]
    weights = runtime["block_linear_weights"]
    known = set(config["strong_baseline_by_depth"].values())
    for metric, entries in data.items():
        verdicts[metric] = {}
        for name, row in entries.items():
            pred, _, _ = vectors(rows, name, metric)
            verdicts[metric][name] = pred
            depth = loops(name)
            strong = config["strong_baseline_by_depth"][str(depth)]
            base = vectors(rows, strong, metric)[0] == gold
            row["vs_validation_selected_same_depth"] = dict(reference=strong, **paired(base, pred == gold))
            head_count = 2 if name.startswith(("loopcd_fixed", "loopcd_adaptive")) else 1
            row["token_layers"] = int(work.sum()) * runtime["model_config_layers"] * depth
            row["total_head_positions"] = int(heads.sum()) * head_count
            row["linear_flop_proxy"] = 2 * (int(work.sum()) * weights * depth + int(heads.sum()) * h * v * head_count)
            row["correct_per_linear_pf"] = row["correct"] / (row["linear_flop_proxy"] / 1e15)
            row["mean_executed_loops"] = depth
            traces = [r for r in latency["observations"] if r["method"] == name]
            row["cost"] = {
                "prefill_p50_ms": float(np.median([r["prefill_ms"] for r in traces])),
                "prefill_p95_ms": float(np.quantile([r["prefill_ms"] for r in traces], .95)),
                "decode_p50_ms": float(np.median([r["decode_ms"] for r in traces])),
                "decode_p95_ms": float(np.quantile([r["decode_ms"] for r in traces], .95)),
                "prefill_plus_decode_p50_ms": float(np.median([r["prefill_ms"] + r["decode_ms"] for r in traces])),
                "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in traces),
            }
        pred = np.where(shallow, verdicts[metric][route["shallow"]], verdicts[metric][route["deep"]])
        verdicts[metric]["adaptive_depth_replay"] = pred
        routing[metric] = dict(n=len(rows), correct=int((pred == gold).sum()),
                               accuracy_pct=100 * float(np.mean(pred == gold)),
                               mean_loop_budget=float(depths.mean()), exit_counts=dict(Counter(depths.tolist())),
                               token_weighted_loop_budget=float(np.average(depths, weights=work)),
                               deep_reference=route["deep"],
                               vs_deep=paired(verdicts[metric][route["deep"]] == gold, pred == gold))

    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        cols = [(metric, name, preds) for metric, entries in verdicts.items() for name, preds in entries.items()]
        writer.writerow(["id", "gold", "adaptive_replay_loops"] + [f"{m}:{n}" for m, n, _ in cols])
        for idx, row in enumerate(rows):
            writer.writerow([row["id"], row["labels"][row["gold"]], depths[idx]]
                            + [row["labels"][int(p[idx])] for _, _, p in cols])

    dump(out / "test_metrics.json", data)
    dump(out / "adaptive_depth_replay.json", routing)
    selected = set(known) | {"native_k1", "native_k2", "native_k3", "native_k4", "secant_k3", "secant_k4"}
    for metric, entries in data.items():
        print(metric)
        for name in entries:
            if name in selected:
                print(name, entries[name]["correct"], entries[name]["accuracy_pct"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "analyze"))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    (freeze if args.command == "freeze" else analyze)(args.out)
