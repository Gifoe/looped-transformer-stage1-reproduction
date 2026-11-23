"""Project saved controllers onto independent native functional matrices on CPU."""

import argparse
from collections import defaultdict
from hashlib import file_digest
import json
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def group(path):
    dataset, stage, depth, anchor = path.stem.split("__")
    return dataset, stage, int(depth[1:]), int(anchor[1:])


def factors(path):
    model_state = torch.load(path, map_location="cpu", weights_only=True)["state"]
    u, w = (model_state[name].flatten().double() for name in ("out.weight", "encode.weight"))
    if (u.numel(), w.numel()) != (2048, 2048) or min(u.norm(), w.norm()) <= 0:
        raise ValueError(f"Invalid factors for {path}")
    return u, w


def project(fit, holdout, arms):
    names = sorted(arms)
    factors_by_arm = [factors(arms[name]) for name in names]
    u, w = (torch.stack(items) for items in zip(*factors_by_arm, strict=True))
    strength = u.norm(dim=1) * w.norm(dim=1)
    u, w = torch.nn.functional.normalize(u), torch.nn.functional.normalize(w)
    cosine = (u @ u.T) * (w @ w.T)
    scales = {group(path)[:3]: read(path)["sigma"] for path in fit.glob("*.json")}
    projections = defaultdict(list)
    receipts = {}
    for receipt in sorted(holdout.glob("*.json")):
        dataset, stage, depth, anchor = group(receipt)
        record = read(receipt)
        matrix_path = receipt.with_suffix(".pt")
        if digest(matrix_path) != record["matrix_sha256"] or anchor != depth - 1:
            raise ValueError(f"Invalid native boundary or matrix hash at {receipt.name}")
        data = torch.load(matrix_path, map_location="cpu", weights_only=True)
        sigma = scales[dataset, stage, min(depth, 4)]
        slopes = ((data["matrix"].double() @ w.T).T * u).sum(1) / sigma
        receipts[receipt.stem] = record["matrix_sha256"]
        for i, name in enumerate(names):
            projections[name].append(dict(dataset=dataset, stage=stage, depth=depth, n=data["n"],
                                          sigma=sigma, unit_slope=float(slopes[i]),
                                          native_tangent=float(slopes[i] * strength[i])))
    if len(receipts) != 45:
        raise ValueError(f"Expected 45 saved function matrices, found {len(receipts)}")
    return {name: dict(sha256=digest(arms[name]), norm=float(strength[i]), cells=projections[name],
                       operator_cosine={other: float(cosine[i, j]) for j, other in enumerate(names)})
            for i, name in enumerate(names)}, receipts


def join_development(projections, paired):
    """Report paired outcomes alongside slopes on a distinct, fixed-prefix cohort."""
    report = {}
    for name, arm in projections.items():
        slices = {}
        for depth in (2, 3, 4):
            key = f"{depth}"
            cells = [x for x in arm["cells"] if x["depth"] == depth]
            math = {x["stage"]: x["unit_slope"] for x in cells if x["dataset"] == "gsm8k"}
            candidate = {x["dataset"]: x["unit_slope"] for x in cells if x["stage"] == "candidate"}
            development = paired[f"gsm8k/k{depth}/gsm_process"].get(name)
            summary = None if development is None else {
                metric: dict(count=development["metrics"][metric]["count"], n=development["metrics"][metric]["n"],
                             versus_native=development["paired"]["native"][metric],
                             versus_mixed=development["paired"]["mixed"][metric])
                for metric in ("process_correct", "premature_eos")}
            slices[key] = dict(directional_math=math, directional_candidate=candidate,
                               positive_groups=sum(x["unit_slope"] > 0 for x in cells), total_groups=len(cells),
                               development=summary)
        report[name] = slices
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("fit", "holdout", "weights", "paired", "development", "out"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--historical", action="append", default=[], metavar="ARM=WEIGHT")
    args = parser.parse_args()
    torch.set_num_threads(4)
    arms = {path.parent.name: path for path in args.weights.glob("*/controller.pt")}
    for item in args.historical:
        name, filename = item.split("=", 1)
        arms[name] = Path(filename)
    results, matrices = project(args.fit, args.holdout, arms)
    development = read(args.paired)["results"]
    for key, cells in development.items():
        dataset, depth, mode = key.split("/")
        for name in set(cells) - {"native"}:
            receipt = read(args.development / f"{dataset}__{name}__{depth}__{mode}.json")
            if results[name]["sha256"] != receipt["plan"]["checkpoint_sha256"]:
                raise ValueError(f"Wrong saved checkpoint {name} at {key}")
    joined = join_development(results, development)
    output = dict(controllers=results, matrix_sha256=matrices, paired_development_sha256=digest(args.paired),
                  local_slopes_and_heldout_trajectory_outcomes=joined)
    temporary = args.out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    for name, row in joined.items():
        print(name, "norm", round(results[name]["norm"], 3), "K2–K4 positive",
              sum(x["positive_groups"] for x in row.values()),
              "cos vs mixed", round(results[name]["operator_cosine"]["mixed"], 3), flush=True)
        for depth, cell in row.items():
            observed = cell["development"]
            print("K" + depth, "math unit slopes", {k: round(v, 3) for k, v in cell["directional_math"].items()},
                  "free joint and early EOS", [x["count"] for x in observed.values()] if observed else None, flush=True)


if __name__ == "__main__":
    main()
