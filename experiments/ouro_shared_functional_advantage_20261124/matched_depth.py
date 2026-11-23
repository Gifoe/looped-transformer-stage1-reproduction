"""Measure depth differences on a common test subset, including the K4 full-cohort arm."""

import argparse
import json
from pathlib import Path

import minimal
from paired_analysis import compare
from summarize_evaluation import checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for label in ("protocol", "results", "out"):
        p.add_argument(f"--{label}", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.protocol.read_text())
    saved, _ = checked(a.results, a.protocol, False)
    arm = plan["selected"]
    pairs = sorted({(case["dataset"], case["mode"]) for case in plan["cases"]})
    result = {}
    for ds, mode in pairs:
        rows = {}
        for label in plan["transfer_arms"]:
            for depth in (2, 3, 4, 5, 6):
                key = f"{ds}__{label}__k{depth}__{mode}"
                values = saved[key][0]
                rows[(label, depth)] = {r["id"]: r for r in values}
        shared = sorted(set.intersection(*(set(group) for group in rows.values())))
        name = f"{ds}/{mode}"
        result[name] = dict(ids=shared, depths={})
        for k in (2, 3, 4, 5, 6):
            current = [rows[arm, k][key] for key in shared]
            result[name]["depths"][f"k{k}"] = {
                label: compare([rows[label, k][key] for key in shared], current)
                for label in plan["transfer_arms"] if label != arm
            }
    minimal.dump(a.out, result)
    print({key: len(value["ids"]) for key, value in result.items()}, flush=True)


if __name__ == "__main__":
    main()
