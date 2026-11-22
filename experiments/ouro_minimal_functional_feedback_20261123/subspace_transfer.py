"""Validate cross-depth functional gradient projections without running Ouro."""

import argparse
from pathlib import Path

import numpy as np

from subspace import DEPTHS, RANKS, SEED, dump, load, random_basis, sha


def energy_interval(x, basis, indexes):
    centered = x - x.mean(axis=0)
    norms = np.square(centered).sum(axis=-1)
    projected = np.square(centered @ basis).sum(axis=-1)
    ratio = projected[indexes].sum(axis=1) / norms[indexes].sum(axis=1).clip(min=1e-20)
    return dict(energy_fraction=float(projected.sum() / norms.sum()),
                ci95=np.quantile(ratio, [.025, .975]).tolist())


def transfer(folder, split, basis):
    data = {f'k{k}': load(folder / f'{split}_k{k}.pt')['values'] for k in DEPTHS}
    result = {}
    for kind in ('shared', 'prefix'):
        result[kind] = {}
        for target, values in data.items():
            x = values[kind].numpy().astype(np.float64)
            draws = np.random.default_rng(SEED).integers(0, len(x), (10000, len(x)))
            tested = {depth: arrays['centered'].numpy() for depth, arrays in basis[kind].items()}
            tested['random'] = random_basis(RANKS[-1])
            tested['hidden_pca_target_depth'] = basis['hidden'][target]['centered'].numpy()
            result[kind][target] = {name: {str(r): energy_interval(x, q[:, :r], draws) for r in RANKS}
                                    for name, q in tested.items()}
    return dict(n=len(next(iter(data.values()))['shared']), projections=result,
                input_sha256={key: sha(folder / f'{split}_{key}.pt') for key in data})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--external', action='store_true')
    args = parser.parse_args()
    path = args.folder / 'arc_basis.pt'
    fit = load(path)
    if fit['train_ids_sha256'] != sha(args.folder / 'ARC_GRADIENT_LOCK.json'):
        raise ValueError('ARC basis cohort hash mismatch')
    split = 'external' if args.external else 'validation'
    result = transfer(args.folder, split, fit['spaces'])
    result.update(arc_basis_sha256=sha(path), analysis_sha256=sha(Path(__file__)),
                  intervals='10,000 question bootstrap resamples with the full held-out cohort centering fixed. Intervals are descriptive and unadjusted.')
    name = ('OPENBOOKQA' if args.external else 'ARC') + '_CROSS_DEPTH_GRADIENT_TRANSFER.json'
    dump(args.folder / name, result)


if __name__ == '__main__':
    main()
