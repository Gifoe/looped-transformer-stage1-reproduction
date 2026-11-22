"""Measure realized prefix edit directions using existing, frozen raw state captures."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from minimal import DEPTHS, dump, sha


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(states, v, u, bound, *, donor=None):
    x = (states @ v.T) @ u.T
    size = np.linalg.norm(x, axis=1)
    native = np.linalg.norm(states, axis=1)
    if (size == 0).any() or (native == 0).any():
        raise ValueError("Degenerate evaluated prefix state or edit")
    unit = x / size[:, None]
    singular = np.linalg.svd(unit, compute_uv=False)
    pair = unit @ unit.T
    cosine = pair[np.triu_indices(len(states), k=1)]
    before = size / native
    out = dict(
        n=len(states),
        cap_fraction=float((before >= bound).mean()),
        raw_l2_ratio_quantiles=np.quantile(before, [.1, .5, .9]).tolist(),
        effective_l2_ratio_quantiles=np.quantile(np.minimum(before, bound), [.1, .5, .9]).tolist(),
        unit_edit_energy_first_pc=float(singular[0]**2 / len(unit)),
        all_distinct_question_cosine_quantiles=np.quantile(cosine, [.1, .5, .9]).tolist(),
    )
    if donor is not None:
        out["matched_donor_cosine_quantiles"] = np.quantile(pair[np.arange(len(unit)), donor], [.1,.5,.9]).tolist()
    return out


def run(root):
    lock = read(root / "FREEZE.json")
    arm = lock["arms"][lock["best"]]
    checkpoint = root / arm["file"]
    if sha(checkpoint) != arm["sha256"]:
        raise ValueError("Geometry expects the exact frozen module")
    model = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if model["arch"]["kind"] != "linear":
        raise ValueError("This audit expects the locked linear architecture")
    v = model["state"]["encode.weight"].numpy().astype(np.float64)
    u = model["state"]["out.weight"].numpy().astype(np.float64)
    sub = root / "SUBSPACE_ANALYSIS"
    donor_map = read(root / "CAUSAL_ANALYSIS" / "B_FACTORIAL.json")["donor_question"]
    out = {}
    for split, manifest in (("train", "ARC_GRADIENT_LOCK.json"), ("validation", "ARC_GRADIENT_LOCK.json"),
                            ("external", "OPENBOOKQA_GRADIENT_LOCK.json")):
        receipts = read(sub / manifest)
        ids = receipts["cohorts"][split]["ids"]
        donor = None
        if split == "validation":
            indexes = {name: i for i, name in enumerate(ids)}
            donor = [indexes[donor_map[q]] for q in ids]
        out[split] = {}
        for k in DEPTHS:
            path = sub / f"{split}_k{k}.pt"
            obj = torch.load(path, map_location="cpu", weights_only=True)
            assert obj["cohort_lock_sha256"] == sha(sub / manifest)
            h = obj["values"]["hidden"].numpy().astype(np.float64)
            assert len(h) == len(ids)
            out[split][f"k{k}"] = dict(
                prefix_summary=summarize(h, v, u, model["arch"]["bound"], donor=donor),
                source_sha256=sha(path),
            )
    dump(sub / "EFFECTIVE_UPDATE_GEOMETRY.json", dict(
        scope="CPU replay of the frozen linear module at final prompt positions. Candidates share the prefix. "
              "Cosines characterize output directions; the donor intervention additionally writes at later answer predictors.",
        weights_sha256=arm["sha256"],
        depth_stats=out,
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    run(p.parse_args().root)
