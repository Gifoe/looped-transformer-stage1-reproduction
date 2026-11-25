"""Identify how gold probability changes the candidate training gradient."""

import argparse
import json
import math
from pathlib import Path
import statistics

from scipy.special import expit


def rows(path):
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def align(probe, alternatives):
    value = probe["slopes"]["t1"]
    rival_count = math.log(alternatives - 1)
    direct_mult = float(expit(rival_count - probe["q_direct"]))
    loop_mult = float(expit(rival_count - probe["q_loop"]))
    direct = direct_mult * value["direct"]
    loop = loop_mult * value["loop"]
    return dict(direct=direct, loop=loop, excess=loop - direct), loop_mult


def run(a):
    output = {}
    for name in a.sources:
        choices = {r["sample_id"]: len(r["candidate"]["choices"]["text"])
                   for r in rows(a.pools / f"{name}.jsonl")}
        summary = []
        for fold in sorted((a.moments / name / "candidate/k4").glob("fold_*.jsonl")):
            for r in rows(fold):
                corrected, level = align(r, choices[r["id"]])
                original = r["slopes"]["t1"]
                summary.append((original["loop"], original["excess"], corrected["loop"], corrected["excess"], level))
        if len(summary) != len(choices):
            raise ValueError(f"Incomplete slopes at {name}")
        metrics = ("q_loop", "q_excess", "normalized_ce_loop", "normalized_ce_excess", "mean_rival_probability")
        output[name] = {"n": len(summary), "means": dict(zip(
            metrics, (statistics.mean(x) for x in zip(*summary)), strict=True))}
    a.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("pools", "moments", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--sources", nargs="+", required=True)
    run(p.parse_args())
