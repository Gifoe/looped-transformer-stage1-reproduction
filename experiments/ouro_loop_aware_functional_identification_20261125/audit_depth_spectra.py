"""Measure across-depth alignment on the exact same full ARC training population."""

import argparse
from itertools import permutations
import json
from pathlib import Path

import torch

from freeze_forecast import top
import minimal


def summarize(a):
    torch.set_num_threads(4)
    spectra, operators, locks = {}, {}, []
    for depth in (2, 3, 4):
        root = a.input / f"k{depth}"
        config = json.loads((root / "LOCK.json").read_text())
        locks.append(config["source_sha256"])
        files = sorted(root.glob("fold_*.pt"))
        folds = [torch.load(f, weights_only=True, map_location="cpu") for f in files]
        n = sum(f["count"] for f in folds)
        if len(folds) != 4 or n != 1119:
            raise ValueError(f"Expected full ARC population and four folds at K{depth}")
        ordinary = sum(f["moments"]["loop"] for f in folds) / n
        direct = sum(f["moments"]["direct"] for f in folds) / n
        spectra[str(depth)] = {}
        operators[depth] = {}
        for k, matrix in (("ordinary", ordinary), ("contrast", ordinary - direct)):
            u, s, w = top(matrix, 16, a.seed)
            stable = float(matrix.square().sum() / s[0].square())
            spectra[str(depth)][k] = {"stable_rank": stable, "singular_values": s.tolist(), "rank1_energy": 1./stable}
            operators[depth][k] = (matrix, u, w)
    if len(set(locks)) != 1:
        raise ValueError("Depths do not contain the same locked ARC data")
    pairs = []
    for a_depth, b_depth in permutations(operators, 2):
        for key, (_, u, w) in operators[a_depth].items():
            target, ut, wt = operators[b_depth][key]
            pairs.append(dict(source=a_depth, target=b_depth, operator=key,
                              rank1_cosine=float((u[:, 0] @ ut[:, 0]) * (w[:, 0] @ wt[:, 0])),
                              write_top8_energy=float((u[:, :8].T @ ut[:, :8]).square().sum()/8),
                              read_top8_energy=float((w[:, :8].T @ wt[:, :8]).square().sum()/8),
                              transferred_slope=float(u[:, 0] @ (target @ w[:, 0]))))
    minimal.dump(a.out, dict(input_sha256=locks[0], cohort=1119, spectra=spectra, pairs=pairs, seed=a.seed))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=261125137)
    summarize(p.parse_args())
