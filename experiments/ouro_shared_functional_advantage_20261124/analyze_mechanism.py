"""Summarize finite-dose response, sign errors and paired transport contrasts."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from task_data import load
from trajectory_audit import digest, percentile_interval


def scale_lookup(path):
    scales = {}
    for p in path.glob("*.json"):
        ds, role, k, _ = p.stem.split("__")
        scales[f"{ds}/{role}", int(k[1:])] = json.loads(p.read_text())["sigma"]
    return scales


def rows(directory):
    for path in sorted(directory.glob("*.json")):
        raw = path.with_suffix(".jsonl")
        receipt = json.loads(path.read_text())
        if receipt["rows_sha256"] != digest(raw):
            raise ValueError(f"Replay fingerprint mismatch {raw}")
        data = load(raw)
        if len(data) != receipt["n"]:
            raise ValueError(f"Replay count mismatch {raw}")
        yield from data


def normalized(record, scale):
    g = (record["q_edited"] - record["q_native"]) / scale
    return record["id"], g, record["linear"] / scale, record["write_ratio"], record["q_native"]


def collect(directory, scales, only_k4=False):
    results = defaultdict(list)
    for r in rows(directory):
        if only_k4 and r["depth"] != 4:
            continue
        scale = scales[r["group"], min(r["depth"], 4)]
        key = r["group"], r["depth"], r["arm"], r["dose"]
        results[key].append(normalized(r, scale))
    return results


def summarize(grouped, rng):
    report = {}
    for (group, depth, arm, dose), points in grouped.items():
        if len(points) != len({point[0] for point in points}):
            raise ValueError(f"Repeated evaluation key {group}/{depth}/{arm}/{dose}")
        real, tangent, physical, _ = np.asarray([x[1:] for x in points]).T
        error = abs(real - tangent)
        label = f"{group}/k{depth}/{arm}/{dose:.2f}"
        report[label] = dict(n=len(real), actual=float(real.mean()), linear=float(tangent.mean()),
                             ci95=percentile_interval(real, rng), mae=float(error.mean()),
                             sign_agreement=float((np.sign(real) == np.sign(tangent)).mean()),
                             measured_rms_ratio=float(physical.mean()))
    return report


def transport(early, late, rng):
    report = {}
    for key, first in early.items():
        later = late[key]
        order = {x[0]: x[1:] for x in later}
        if {x[0] for x in first} != set(order):
            raise ValueError(f"Transport pairing differs {key}")
        a = np.asarray([x[1:] for x in first])
        b = np.asarray([order[x[0]] for x in first])
        real, slope, physical, q0diff = (a-b).T
        g, depth, arm, dose = key
        report[f"{g}/k{depth}/{arm}/{dose:.2f}"] = dict(
            n=len(a), early_minus_late=float(real.mean()), adjoint_contrast=float(slope.mean()),
            ci95=percentile_interval(real, rng), mean_realized_ratio_difference=float(physical.mean()),
            max_native_utility_drift=float(abs(q0diff).max()),
        )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("fit", "late", "early", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    a = p.parse_args()
    scales = scale_lookup(a.fit)
    late = collect(a.late, scales)
    early = collect(a.early, scales, only_k4=True)
    rng = np.random.default_rng(261124173)
    report = dict(finite=summarize(late, rng), transport=transport(early, late, rng))
    a.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(len(report["finite"]), "finite cells", len(report["transport"]), "paired transport cells", flush=True)


if __name__ == "__main__":
    main()
