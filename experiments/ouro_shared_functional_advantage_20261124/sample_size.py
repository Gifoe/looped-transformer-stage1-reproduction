"""Test how rapidly candidate-only directions stabilize on nested fitting sets."""

import argparse
import json
from pathlib import Path

import torch

from shared_direction import analyze
from validate_direction import analyze as validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "fit", "holdout", "out"):
        p.add_argument(f"--{name}", required=True, type=Path)
    a = p.parse_args()
    arguments = (6, 261124173, 16, 500)
    controls = {}
    for n in (8, 16, 32, 48):
        folder = a.results / f"SAMPLE_MOMENTS_{n}" if n < 48 else a.fit
        records = {}
        for path in sorted(folder.glob("*__candidate__*.pt")):
            group = torch.load(path, map_location="cpu", weights_only=True)
            full = json.loads((a.fit / (path.stem + ".json")).read_text())
            # Keep the same full-fold unit convention across the data-size ablation.
            group["sigma"] = full["sigma"]
            records[path.stem] = group
        fit, axes = analyze(records, *arguments)
        direction = {name: value.float() for name, value in zip(("u", "w"), axes["worst_group"], strict=True)}
        controls[f"n{n}"] = {"fit": fit, "direction": direction}
    result = validate(a.fit, a.holdout, {label: row["direction"] for label, row in controls.items()})
    for label, row in result.items():
        row["fit"] = controls[label]["fit"]
        for kind in ("candidate", "generation"):
            groups = [x["value"] for x in row["cells"] if (x["role"] == "candidate") == (kind == "candidate")]
            row[kind] = dict(n=len(groups), positives=sum(v>0 for v in groups), weakest=min(groups), mean=sum(groups)/len(groups))
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    torch.save({k: v["direction"] for k, v in controls.items()}, a.out.with_suffix(".pt"))
    print({k: {x: row[x] for x in ("candidate", "generation")} for k, row in result.items()}, flush=True)


if __name__ == "__main__":
    main()
