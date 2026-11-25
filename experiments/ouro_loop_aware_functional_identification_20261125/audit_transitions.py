"""Summarize native correctness transitions on complete candidate sources."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


LABELS = {(0, 0): "persistent_failure", (0, 1): "rescued",
          (1, 0): "degraded", (1, 1): "always_correct"}


def inspect(folder):
    cells = defaultdict(list)
    for file in sorted(folder.glob("fold_*.jsonl")):
        with file.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                left, right = row["native_direct"], row["native_loop"]
                if left not in (0, 1) or right not in (0, 1):
                    raise ValueError("Correctness transitions require candidate scoring")
                label = LABELS[(int(left), int(right))]
                cells[label].append(row)
    if not cells:
        raise ValueError(f"No complete candidate outcomes under {folder}")
    results = {}
    for label, rows in cells.items():
        slopes = {}
        for method in ("t1", "t2", "mixed"):
            slopes[method] = {axis: statistics.mean(r["slopes"][method][axis] for r in rows)
                              for axis in ("direct", "loop", "excess")}
        results[label] = {"n": len(rows),
                          "mean_native_utility_change": statistics.mean(r["q_loop"] - r["q_direct"] for r in rows),
                          "mean_capped_direction_slopes": slopes}
    return results


def binary_polarity(folder, pool):
    with pool.open(encoding="utf-8") as source:
        answers = {r["sample_id"]: r["answer_content"].casefold() for r in map(json.loads, source)}
    cells = defaultdict(list)
    for file in sorted(folder.glob("fold_*.jsonl")):
        with file.open(encoding="utf-8") as source:
            for row in map(json.loads, source):
                label = answers[row["id"]]
                cells[label, int(row["native_loop"])].append(row)
    if set(label for label, _ in cells) != {"yes", "no"}:
        raise ValueError("Expected a Yes/No population")
    return {f"{label}/{correct}": dict(n=len(rows), slopes={
                axis: statistics.mean(r["slopes"]["t1"][axis] for r in rows)
                for axis in ("loop", "direct", "excess")})
            for (label, correct), rows in sorted(cells.items())}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moments", required=True, type=Path)
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--boolq-pool", type=Path)
    p.add_argument("--boolq-out", type=Path)
    a = p.parse_args()
    out = {name: inspect(a.moments / name / "candidate/k4") for name in a.sources}
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if a.boolq_pool is not None:
        polarity = binary_polarity(a.moments / "boolq/candidate/k4", a.boolq_pool)
        a.boolq_out.write_text(json.dumps(polarity, indent=2) + "\n", encoding="utf-8")
