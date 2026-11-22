"""Fit matched rank-eight random-subspace initializations on ARC Train only."""

import argparse
from pathlib import Path

import torch

from minimal import runtime, save_net
from subspace import SEED, dump, load, random_basis, read, sha
from train import count, evaluate, fit, partitions


def run(args):
    target = args.out / 'SUBSPACE_ANALYSIS'
    fixed_source = target / 'arc_basis.pt'
    chosen = load(fixed_source)
    if chosen['rank'] != 8 or chosen['train_ids_sha256'] != sha(target / 'ARC_GRADIENT_LOCK.json'):
        raise ValueError('Expected the hashed ARC-Train rank-eight functional basis')
    training = read(args.out / 'VALIDATION.json')
    if not {'subspace_joint_t1', 'subspace_fixed_t1', 'linear8_t1'}.issubset(training):
        raise ValueError('Run the trained-gradient and zero-initialized controls first')
    old, model, tok, env = runtime(args.archive, args.previous, args.model)
    train, dev = [old.read_data(args.corpus, split) for split in ('train', 'validation')]
    groups, metadata = partitions(old, tok, train)
    basis = torch.tensor(random_basis(8), dtype=torch.float32)
    matches = {}
    for kind, fixed in (('joint', False), ('fixed', True)):
        arm = f'subspace_random_{kind}_t1'
        counterpart = f'subspace_{kind}_t1'
        dose = training[counterpart]['dose']
        net, timing = fit(old, model, groups, metadata, None, 'linear', 8, 't1', basis, fixed)
        rows, cost = evaluate(old, model, tok, dev, net, dose)
        checkpoint = target / f'{arm}.pt'
        save_net(checkpoint, net, timing)
        file = target / f'VAL_{arm}.jsonl'
        tmp = file.with_suffix('.tmp')
        old.write_predictions(tmp, rows)
        tmp.replace(file)
        measured = count(rows)
        matches[arm] = dict(fit=timing, validation_cost=cost, checkpoint=checkpoint.name,
                            checkpoint_sha256=sha(checkpoint), scores_sha256=sha(file), random_control=measured,
                            gradient_initialized=training[counterpart]['counts'], dose=dose)
        print(arm, measured, flush=True)
    report = dict(comparisons=matches, frozen_runtime=env, seed=SEED, rank=8,
                  train_sha256=sha(args.corpus / 'train.parquet'), validation_sha256=sha(args.corpus / 'validation.parquet'),
                  gradient_basis_sha256=sha(fixed_source), source_sha256=sha(Path(__file__)),
                  initialization='A seeded Gaussian orthonormal basis replaces the ARC gradient basis. Encoder scaling, training schedule, data, doses and architectures match the paired fits.')
    dump(target / 'SUBSPACE_RANDOM_INIT_CONTROLS.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('model', 'archive', 'previous', 'corpus', 'out'):
        parser.add_argument(f'--{flag}', required=True, type=Path)
    run(parser.parse_args())
