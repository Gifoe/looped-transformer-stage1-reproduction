"""Evaluate independently collected matrix signs using only fitting-fold scales."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn.functional import normalize


def key(path):
    ds, role, depth, _ = path.stem.split("__")
    return ds, role, int(depth[1:])


def vectors(path):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    state = raw["state"]
    u = state["out.weight"].flatten()
    w = state["encode.weight"].flatten()
    return {"u": normalize(u, dim=0), "w": normalize(w, dim=0)}


def analyze(train, testing, arms):
    scales = {key(path): json.loads(path.read_text())["sigma"] for path in train.glob("*.json")}
    paths = sorted(testing.glob("*.pt"))
    observations = {name: [] for name in arms}
    for path in paths:
        ds, role, k = key(path)
        fit_k = min(k, 4)
        scale = scales[ds, role, fit_k]
        row = torch.load(path, map_location="cpu", weights_only=True)
        for name, pair in arms.items():
            value = float(pair["u"] @ (row["matrix"] @ pair["w"])) / scale
            observations[name].append(dict(dataset=ds, role=role, depth=k, anchor=row["anchor"], value=value,
                                           n=row["n"], sigma=scale, scale_depth=fit_k))
    report = {}
    for arm, cells in observations.items():
        panels = {"trained_depths": [row for row in cells if row["depth"] < 5],
                  "untrained_depths": [row for row in cells if row["depth"] >= 5]}
        summary = {name: {"min": min(x["value"] for x in selection),
                          "mean": sum(x["value"] for x in selection)/len(selection),
                          "positive": sum(x["value"] > 0 for x in selection), "n": len(selection)}
                   for name, selection in panels.items() if selection}
        report[arm] = dict(summary=summary, cells=cells)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "holdout", "shared", "candidate", "rank_only", "retention", "multitask", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    arms = torch.load(a.shared, map_location="cpu", weights_only=True)
    arms["candidate_direction"] = torch.load(a.candidate, map_location="cpu", weights_only=True)["worst_group"]
    arms.update({name: vectors(getattr(a, name)) for name in ("rank_only", "retention", "multitask")})
    result = analyze(a.train, a.holdout, arms)
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print({arm: row["summary"] for arm, row in result.items()}, flush=True)


if __name__ == "__main__":
    main()
