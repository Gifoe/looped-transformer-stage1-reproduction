"""Measure ordinary and loop-specific ARC functional response across Ouro versions."""

import argparse
from itertools import permutations
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from evaluate_transfer import load_runtime
import minimal
from collect_identification import collect
import collect_identification
import finite_identification
from freeze_forecast import top


def collect_version(a):
    sys.path.insert(0, str(a.previous))
    from arc_features import archive

    a.out.mkdir(parents=True, exist_ok=True)
    inputs = dict(models_sha256=minimal.sha(a.models), model_receipt_sha256=minimal.sha(a.model / "MODEL_LOCK.json"),
                  source_sha256=minimal.sha(a.train), dev_sha256=minimal.sha(a.validation),
                  checkpoint_sha256=minimal.sha(a.t1), code_sha256=minimal.sha(Path(__file__)),
                  legacy_moment_sha256=minimal.sha(collect_identification.__file__),
                  legacy_finite_sha256=minimal.sha(finite_identification.__file__))
    lock = a.out / "MECHANISM_LOCK.json"
    if lock.exists() and json.loads(lock.read_text()) != inputs:
        raise ValueError("Cross-model mechanism inputs changed after the first measurement")
    minimal.dump(lock, inputs)

    model, tokenizer, _, _ = load_runtime(a)
    old = archive(a.archive)
    bridge = SimpleNamespace(sha=minimal.sha, dump=minimal.dump,
                             runtime=lambda *_: (old, model, tokenizer, {"model_sha256": inputs["model_receipt_sha256"]}))
    collect_identification.minimal = bridge
    finite_identification.minimal = bridge
    shared = dict(archive=a.archive, previous=a.previous, model=a.model, depth=4,
                  mode="candidate", dataset="arc_challenge", operator=[f"t1={a.t1}"], batch=4)
    collect(SimpleNamespace(**shared, input=a.train, out=a.out / "train", folds=4,
                            limit=None, token_budget=2304, checkpoint_batches=24, full_context=False))
    finite_lock = a.out / "finite" / "FINITE.json"
    if finite_lock.exists():
        prior = json.loads(finite_lock.read_text())
        if prior["finite_sha256"] != minimal.sha(finite_lock.with_name("finite.jsonl")):
            raise ValueError("The archived finite-dose rows were changed")
    else:
        finite_identification.measure(SimpleNamespace(**shared, input=a.validation, limit=24, out=finite_lock.parent))


def summarize_group(path, net):
    lock = json.loads((path / "LOCK.json").read_text())
    folds = [torch.load(path / f"fold_{i}.pt", weights_only=True, map_location="cpu") for i in range(4)]
    ids = [record["id"] for fold in folds for record in fold["records"]]
    n = len(ids)
    if n != 1119 or len(set(ids)) != n or sum(fold["count"] for fold in folds) != n:
        raise ValueError(f"Incomplete and unique ARC training cohort required at {path}")
    if lock["operator_sha256"]["t1"] != net["sha"]:
        raise ValueError("The measured functional slopes used a different controller")
    ordinary = sum(fold["moments"]["loop"] for fold in folds) / n
    direct = sum(fold["moments"]["direct"] for fold in folds) / n
    result, axes = {}, {}
    for name, matrix in (("ordinary", ordinary), ("contrast", ordinary-direct)):
        u, s, w = top(matrix, 16, 261125137)
        stability = []
        for fold in folds:
            fo = fold["moments"]["loop"]
            fd = fold["moments"]["direct"]
            iu, _, iw = top(fo if name == "ordinary" else fo-fd, 8, 261125138)
            stability.append((iu[:, :8], iw[:, :8]))
        overlap = [float(((ou.T @ iu).square().sum() + (ow.T @ iw).square().sum())/16)
                   for i, (ou, ow) in enumerate(stability) for iu, iw in stability[i+1:]]
        slope = float(net["u"] @ matrix @ net["w"])
        result[name] = dict(singular_values=s.tolist(), stable_rank=float(matrix.square().sum()/s[0].square()),
                            fold_top8_overlap=overlap, t1_unclipped_slope=slope)
        axes[name] = (u, w)
    return result, axes, set(ids), lock["source_sha256"]


def finite_scores(path):
    receipt = json.loads((path / "FINITE.json").read_text())
    rows = path / "finite.jsonl"
    if receipt["finite_sha256"] != minimal.sha(rows) or receipt["n"] != 24:
        raise ValueError("The archived finite-dose diagnostic differs")
    records = [row for line in rows.read_text().splitlines() if (row := json.loads(line))["operator"] == "t1"]
    summary = []
    ids = {r["id"] for r in records}
    for dose in receipt["doses"]:
        selected = [row for row in records if row["dose"] == dose]
        if len(selected) != 24 or {r["id"] for r in selected} != ids:
            raise ValueError("Unmatched finite-dose conditions")
        for sign in ("positive", "negative"):
            direct = torch.tensor([r[sign]["direct"] for r in selected], dtype=torch.float64)
            loop = torch.tensor([r[sign]["loop"] for r in selected], dtype=torch.float64)
            changes = torch.stack((direct, loop, loop-direct), dim=1)
            sample = torch.randint(24, (4000, 24), generator=torch.Generator().manual_seed(261125137))
            interval = torch.quantile(changes[sample].mean(1), torch.tensor([.025, .975], dtype=changes.dtype), dim=0)
            summary.append(dict(dose=dose, sign=sign, mean=changes.mean(0).tolist(), ci95=interval.T.tolist()))
    return summary, ids, receipt["source_sha256"]


def summarize(a):
    torch.set_num_threads(4)
    state = torch.load(a.t1, weights_only=True, map_location="cpu")["state"]
    net = {"u": state["out.weight"].float().flatten(), "w": state["encode.weight"].float().flatten(),
           "sha": minimal.sha(a.t1)}
    labels = json.loads(a.models.read_text())
    centers = {"ouro-1.4b-base": (a.base_train, a.base_finite),
               **{name: (a.results / name / "train", a.results / name / "finite") for name in labels}}
    spectra, axes, finite = {}, {}, {}
    for name, (train, dev) in centers.items():
        spectra[name], axes[name], ids, split = summarize_group(train, net)
        finite[name], dev_ids, dev_split = finite_scores(dev)
        if name == "ouro-1.4b-base":
            identity = ids, split, dev_ids, dev_split
        elif (ids, split, dev_ids, dev_split) != identity:
            raise ValueError(f"Unmatched mechanistic ARC populations at {name}")
    overlaps = []
    for source, target in permutations(axes, 2):
        for metric, (u, w) in axes[source].items():
            ut, wt = axes[target][metric]
            overlaps.append(dict(source=source, target=target, moment=metric,
                                 rank1_cosine=float((u[:, 0] @ ut[:, 0]) * (w[:, 0] @ wt[:, 0])),
                                 read_top8_overlap=float((w[:, :8].T @ wt[:, :8]).square().sum()/8),
                                 write_top8_overlap=float((u[:, :8].T @ ut[:, :8]).square().sum()/8)))
    minimal.dump(a.out, dict(train=1119, validation=24, spectra=spectra, overlaps=overlaps, finite=finite,
                             dose_columns=["direct", "loop", "contrast"], train_sha256=identity[1],
                             validation_sha256=identity[3], operator_sha256=net["sha"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parsers = parser.add_subparsers(dest="command", required=True)
    worker = parsers.add_parser("measure")
    for field in ("models", "model", "t1", "archive", "previous", "train", "validation", "out"):
        worker.add_argument(f"--{field}", type=Path, required=True)
    worker.add_argument("--name", required=True)
    group = parsers.add_parser("summarize")
    for field in ("models", "t1", "base-train", "base-finite", "results", "out"):
        group.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    (collect_version if args.command == "measure" else summarize)(args)
