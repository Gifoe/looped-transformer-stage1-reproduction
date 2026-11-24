"""Probe the first output-factor gradient of candidate-only rank-one learning."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn.functional import normalize

from validate_direction import key


def moments(folder):
    for path in sorted(folder.glob("*.pt")):
        yield path, torch.load(path, map_location="cpu", weights_only=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for item in ("fit", "holdout", "out"):
        p.add_argument(f"--{item}", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=256)
    a = p.parse_args()
    full, outer = torch.zeros(2048, 2048), torch.zeros(2048, 2048)
    scales = {}
    for receipt in a.fit.glob("*.json"):
        scales[key(receipt)] = json.loads(receipt.read_text())["sigma"]
    fit = sorted(a.fit.glob("*__candidate__*.pt"))
    for path in fit:
        value = torch.load(path, map_location="cpu", weights_only=True)
        full += value["matrix"] / value["sigma"]
        outer += value["mean_gradient"][:, None] * value["mean_state"][None, :] / value["sigma"]
    w = normalize(torch.randn(a.seeds, 2048, generator=torch.Generator().manual_seed(261124173)), dim=1)
    u = {name: normalize(w @ matrix.T, dim=1)
         for name, matrix in (("full", full), ("mean_outer", outer), ("centered_only", full - outer))}
    values = {name: {"candidate": [], "generation": []} for name in u}
    label = {"candidate": [], "generation": []}
    for path, item in moments(a.holdout):
        ds, role, depth = key(path)
        objective = "candidate" if role == "candidate" else "generation"
        product = w @ (item["matrix"].T / scales[ds, role, min(depth, 4)])
        label[objective].append(f"{ds}/{role}/k{depth}")
        for arm, direction in u.items():
            values[arm][objective].append((product * direction).sum(-1))
    report = {}
    for name, row in values.items():
        result = {}
        for task, columns in row.items():
            tensor = torch.stack(columns, dim=1)
            minimum = tensor.min(1).values
            result[task] = dict(groups=label[task], n_initializations=a.seeds,
                                all_positive=int((minimum > 0).sum()), median_minimum=float(minimum.median()),
                                mean_group_value=float(tensor.mean()), per_random_start=tensor.tolist())
        report[name] = result
    a.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print({name: {task: (obj["all_positive"], obj["median_minimum"]) for task, obj in row.items()}
           for name, row in report.items()}, flush=True)


if __name__ == "__main__":
    main()
