"""Audit the existing candidate margins and greedy answer trajectories on paired IDs."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import re

import numpy as np
from scipy.special import logsumexp

from paired_analysis import compare
from task_data import load


NUMBER_END = re.compile(r"####\s*[-+]?\d[\d,]*(?:\.\d+)?")
ARMS = ("rank_only", "retention_control", "gen_only", "gen_rank", "multitask_full")
TASKS = {"arc_challenge": ("candidate", "content"), "arc_easy": ("candidate", "content"),
         "csqa": ("candidate", "content"), "obqa": ("candidate", "content"),
         "squad": ("span",), "gsm8k": ("gsm_final", "gsm_process")}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_verified(file):
    receipt = json.loads(file.with_suffix(".json").read_text())
    sig = digest(file)
    if receipt["rows_sha256"] != sig:
        raise RuntimeError(f"Damaged confirmation row file {file}")
    rows = load(file)
    if len(rows) != receipt["plan"]["n"] or len({r["id"] for r in rows}) != len(rows):
        raise RuntimeError(f"Mismatched confirmation cohort {file}")
    return rows, sig


def candidate_utility(row):
    scores = 16 * np.asarray(row["scores"]) / np.asarray(row["chars"])
    target = row["gold"]
    return float(scores[target] - logsumexp(np.delete(scores, target)))


def percentile_interval(diff, rng):
    indices = rng.integers(len(diff), size=(4000, len(diff)))
    return np.quantile(diff[indices].mean(-1), [.025, .975]).tolist()


def candidate_audit(before, after, rng):
    old = np.asarray([candidate_utility(r) for r in before])
    new = np.asarray([candidate_utility(r) for r in after])
    diff = new - old
    b = np.asarray([r["metrics"]["acc_norm"] for r in before])
    a = np.asarray([r["metrics"]["acc_norm"] for r in after])
    return dict(native_mean=float(old.mean()), arm_mean=float(new.mean()), mean_gain=float(diff.mean()),
                ci95=percentile_interval(diff, rng),
                positive_on_correct_both=int(((diff > 0) & a & b).sum()),
                positive_without_correct_choice=int(((diff > 0) & ~a).sum()),
                gained_decision=int((a & ~b).sum()), lost_decision=int((b & ~a).sum()))


def output_profile(rows, process):
    receipt = dict(median_tokens=float(np.median([r["answer_tokens"] for r in rows])),
                   stop_counts=dict(Counter(r["stop"] for r in rows)))
    if process:
        receipt["eos_without_final_marker"] = sum(r["stop"] == "eos" and not NUMBER_END.search(r["prediction"]) for r in rows)
        receipt["eos_before_16_tokens"] = sum(r["stop"] == "eos" and r["answer_tokens"] < 16 for r in rows)
    return receipt


def analyze(archive, out):
    report = {}
    rng = np.random.default_rng(261124173)
    for dataset, modes in TASKS.items():
        for mode in modes:
            for k in range(2, 7):
                if mode == "gsm_process" and k != 4:
                    continue
                origin = archive / f"{dataset}__native__k{k}__{mode}.jsonl"
                before, native_sha = read_verified(origin)
                record = dict(n=len(before), native_sha256=native_sha, arms={})
                for arm in ARMS:
                    path = archive / f"{dataset}__{arm}__k{k}__{mode}.jsonl"
                    after, sha = read_verified(path)
                    metrics = compare(before, after)
                    if mode == "candidate":
                        aux = candidate_audit(before, after, rng)
                    else:
                        aux = {"native": output_profile(before, mode == "gsm_process"),
                               "controller": output_profile(after, mode == "gsm_process")}
                    record["arms"][arm] = dict(sha256=sha, paired=metrics, diagnostics=aux)
                report[f"{dataset}/k{k}/{mode}"] = record
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {len(report)} completed archived conditions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.archive, args.out)


if __name__ == "__main__":
    main()
