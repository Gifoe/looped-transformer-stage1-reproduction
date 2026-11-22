"""Measure task-gradient subspaces at Ouro's causal feedback anchors."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F


DEPTHS = (1, 2, 3, 4)
RANKS = (1, 2, 4, 8, 16)
FAMILIES = ('shared', 'prefix', 'hidden')
SEED = 261123149
WIDTH = 2048


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, result):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def save(path, result):
    tmp = path.with_suffix('.tmp')
    torch.save(result, tmp)
    tmp.replace(path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def hash_order(row, stratum):
    return hashlib.sha256(f'{SEED}:{stratum}:{row["id"]}'.encode()).digest()


def dataset(path, n, split):
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    expected = {'train': 1119, 'validation': 299, 'external': 500}[split]
    if len(rows) != expected or len({r['id'] for r in rows}) != expected or not 0 < n <= expected:
        raise ValueError((split, len(rows), expected, n))
    selected = sorted(rows, key=lambda r: hash_order(r, split))[:n]
    return selected, {'sha256': sha(path), 'ids': [r['id'] for r in selected], 'n': n}


def cohorts(args, external=False):
    if external:
        if args.freeze is None or not args.freeze.is_file() or args.external_data is None:
            raise ValueError('External gradients require the final method freeze and canonical OpenBookQA path')
        data = {'external': dataset(args.external_data, args.external_n, 'external')}
        declaration = {'freeze_sha256': sha(args.freeze)}
    else:
        data = {split: dataset(args.corpus / f'{split}.parquet', n, split)
                for split, n in (('train', args.train_n), ('validation', args.validation_n))}
        if set(data['train'][1]['ids']).intersection(data['validation'][1]['ids']):
            raise ValueError('ARC Train and Validation overlap')
        declaration = {}
    declaration.update(seed=SEED, cohorts={name: detail for name, (_, detail) in data.items()},
                       anchors={'k1': 'after physical layer 21 of 24', 'k2-k4': 'before norm of penultimate loop'})
    lock = args.out / ('OPENBOOKQA_GRADIENT_LOCK.json' if external else 'ARC_GRADIENT_LOCK.json')
    if lock.exists():
        if read(lock) != declaration:
            raise ValueError(f'Cohort changed after locking {lock.name}')
    else:
        dump(lock, declaration)
    return data, lock


def read_runtime(archive, model_path):
    sys.path.insert(0, str(archive.resolve()))
    import run_arc as old

    model, tokenizer, runtime = old.load_runtime(model_path)
    model.requires_grad_(False)
    return old, model, tokenizer, runtime


def by_question(old, tok, rows):
    items, info = old.encode(tok, rows)
    questions = [[] for _ in info]
    for item in items:
        questions[item[2]].append(item)
    order = sorted(range(len(info)), key=lambda q: max(len(x[0]) for x in questions[q]))
    return questions, info, order


def candidate_sums(model, final, packed, n):
    logits = model.lm_head(final).float()
    logp = -F.cross_entropy(logits, packed[-1], reduction='none')
    return logp.new_zeros(n).index_add_(0, packed[1], logp)


def batch_loss(sums, rows):
    counts = torch.tensor([len(r['characters']) for r in rows], device=sums.device)
    max_count = max(len(r['characters']) for r in rows)
    mask = torch.arange(max_count, device=sums.device)[None, :] < counts[:, None]
    gold = torch.tensor([r['gold'] for r in rows], device=sums.device)
    chars = sums.new_tensor([x for r in rows for x in r['characters']])
    raw, norm = [sums.new_full(mask.shape, -1e6) for _ in range(2)]
    raw[mask] = sums / 3.
    norm[mask] = 16. * sums / chars
    # Sum the per-question objectives. This preserves each question's scale in a packed batch.
    return .5 * F.cross_entropy(raw, gold, reduction='sum') + .5 * F.cross_entropy(norm, gold, reduction='sum')


def sum_rows(x, owners, n):
    return x.new_zeros((n, x.shape[-1])).index_add_(0, owners, x)


def measures(gradient, anchor, packed, rows):
    group = packed[1]
    counts = torch.tensor([len(r['labels']) for r in rows], device=group.device)
    indices = torch.arange(len(rows), device=group.device)
    owners = torch.repeat_interleave(indices, counts)[group]
    first = torch.cat((group.new_ones(1, dtype=torch.bool), group[1:] != group[:-1]))
    prefix_owner = owners[first]
    # Adding a shared vector at the listed positions gives exactly the sum of the local derivatives.
    shared = sum_rows(gradient, owners, len(rows))
    prefix = sum_rows(gradient[first], prefix_owner, len(rows))
    hidden = sum_rows(anchor[first], prefix_owner, len(rows)) / counts[:, None]
    local_energy = sum_rows(gradient.square().sum(-1, keepdim=True), owners, len(rows))[:, 0]
    return {'shared': shared, 'prefix': prefix, 'hidden': hidden, 'local_energy': local_energy,
            'token_count': torch.bincount(owners, minlength=len(rows))}


def gradient_rows(args, old, model, tok, depth, rows):
    from minimal import forward

    start = time.perf_counter()
    questions, info, order = by_question(old, tok, rows)
    results = {}
    calls = candidates = padded = scored = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for lo in range(0, len(order), args.batch):
        qids = order[lo:lo + args.batch]
        subset = [x for q in qids for x in questions[q]]
        labels = [info[q] for q in qids]
        packed = old.pack(subset)
        with torch.enable_grad():
            final, anchor = forward(model, packed, depth, leaf=True)
            if not anchor.requires_grad or not anchor.is_leaf:
                raise RuntimeError('The candidate anchor must be a differentiable leaf')
            values = candidate_sums(model, final, packed, len(subset))
            loss = batch_loss(values, labels)
            gradient, = torch.autograd.grad(loss, anchor)
        stats = measures(gradient.float(), anchor.detach().float(), packed, labels)
        for kind, value in stats.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(('Non-finite anchor results', kind, depth, lo))
            if kind not in results:
                results[kind] = torch.empty((len(rows), *value.shape[1:]), dtype=value.dtype, device='cpu')
            results[kind][qids] = value.cpu()
        calls += 1
        candidates += len(subset)
        padded += packed[0].numel() * depth
        scored += packed[-1].numel()
    if not results['local_energy'].any():
        raise RuntimeError('No task gradient reached the intervention state')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError('Ouro parameters did not stay frozen')
    torch.cuda.synchronize()
    return results, dict(seconds=time.perf_counter() - start, questions=len(rows), candidates=candidates,
                         scored_target_tokens=scored, calls=calls, padded_token_loop_uses=padded,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved())


def collect(args, data, lock):
    old, model, tok, runtime = read_runtime(args.archive, args.model)
    cost = {'runtime': runtime, 'parts': {}, 'cohort_lock_sha256': sha(lock)}
    for split, (rows, _) in data.items():
        for k in args.depths:
            tag = f'{split}_k{k}'
            path = args.out / f'{tag}.pt'
            if path.exists():
                obj = load(path)
                if obj['cohort_lock_sha256'] != sha(lock) or obj['source_sha256'] != sha(Path(__file__)):
                    raise ValueError(f'Stale input or implementation for {path.name}')
            else:
                values, run = gradient_rows(args, old, model, tok, k, rows)
                obj = {'values': values, 'cost': run, 'cohort_lock_sha256': sha(lock), 'source_sha256': sha(Path(__file__))}
                save(path, obj)
            cost['parts'][tag] = obj['cost']
            print('gradient', tag, obj['cost'], flush=True)
    dump(args.out / ('OPENBOOKQA_GRADIENT_COST.json' if 'external' in data else 'ARC_GRADIENT_COST.json'), cost)


def decomposition(x):
    """Exact sample-space SVD via the Gram matrix. This never computes a model Jacobian."""
    x = np.asarray(x, dtype=np.float64)
    evals, left = np.linalg.eigh(x @ x.T)
    evals = np.maximum(0., evals[::-1])
    keep = min(RANKS[-1], int((evals > max(1e-20, evals[0] * 1e-13)).sum()))
    if keep == 0:
        raise ValueError('All functional gradient energy is zero')
    selected = evals[:keep]
    basis = (x.T @ left[:, ::-1][:, :keep]) / np.sqrt(selected)
    for column in range(keep):
        pivot = np.abs(basis[:, column]).argmax()
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    energy = float(np.square(x).sum())
    cumulative = {str(r): float(evals[:r].sum() / energy) for r in RANKS}
    used = evals[evals > energy * 1e-15]
    p = used / used.sum()
    return basis, dict(energy=energy, top_energy_fraction=cumulative, top_singular_values=np.sqrt(selected).tolist(),
                       effective_rank_entropy=float(np.exp(-(p * np.log(p)).sum())),
                       effective_rank_participation=float(1. / np.square(p).sum()))


def describe(x):
    mean = x.mean(axis=0, keepdims=True, dtype=np.float64)
    centered = x - mean
    direct, uncentered = decomposition(x)
    central, centered_stats = decomposition(centered)
    own_mean_energy = float(len(x) * np.square(mean).sum())
    return direct, central, mean[0], dict(n=len(x), centered=centered_stats, uncentered=uncentered,
                                        mean_energy_fraction=own_mean_energy / uncentered['energy'])


def float_tensor(x):
    return torch.from_numpy(np.asarray(x, dtype=np.float32).copy())


def load_matrices(args, split):
    files = {f'k{k}': args.out / f'{split}_k{k}.pt' for k in args.depths}
    return {k: load(path)['values'] for k, path in files.items()}


def fit_analysis(args):
    train = load_matrices(args, 'train')
    hold = load_matrices(args, 'validation')
    bases, detail = {}, {}
    for kind in FAMILIES:
        bases[kind], detail[kind] = {}, {}
        for k, values in train.items():
            raw, center, mean, spectra = describe(values[kind].numpy())
            bases[kind][k] = {'uncentered': float_tensor(raw), 'centered': float_tensor(center), 'mean': float_tensor(mean)}
            detail[kind][k] = spectra
        joined = np.concatenate([values[kind].numpy() for values in train.values()])
        raw, center, mean, spectra = describe(joined)
        bases[kind]['joint'] = {'uncentered': float_tensor(raw), 'centered': float_tensor(center), 'mean': float_tensor(mean)}
        detail[kind]['joint'] = spectra
    chosen = bases['shared']['joint']['centered'][:, :args.rank].contiguous()
    if chosen.shape != (WIDTH, args.rank):
        raise ValueError(('Insufficient joint gradient basis rank', chosen.shape, args.rank))
    save(args.out / 'arc_basis.pt', {'basis': chosen, 'spaces': bases, 'rank': args.rank,
                                      'train_ids_sha256': sha(args.out / 'ARC_GRADIENT_LOCK.json'),
                                      'description': 'joint centered SVD, ARC Train only; gradient to one additive vector shared across scored candidate positions'})
    local = {split: {k: {'mean_token_gradient_energy': float(v['local_energy'].sum() / v['token_count'].sum()),
                         'scored_target_tokens': int(v['token_count'].sum())} for k, v in data.items()}
             for split, data in (('train', train), ('validation', hold))}
    dump(args.out / 'FUNCTIONAL_GRADIENT_SPECTRUM.json',
         {'train': detail, 'local_gradient_diagnostics': local, 'cohort_lock_sha256': sha(args.out / 'ARC_GRADIENT_LOCK.json'),
          'basis_sha256': sha(args.out / 'arc_basis.pt'), 'loss': '0.5 CE(raw candidate log-likelihood sums / 3) + 0.5 CE(16 log-likelihood / character count)',
          'gradient_samples': {'shared': 'gradient to one additive vector shared across all scored positions of every candidate in the question',
                               'prefix': 'gradient to one additive vector shared only across last-prompt positions for every candidate',
                               'hidden': 'average native residual at last-prompt position across candidate encodings'}})
    alignment(args, hold, 'validation')


def random_basis(rank):
    draw = np.random.default_rng(SEED).normal(size=(WIDTH, RANKS[-1]))
    return np.linalg.qr(draw)[0][:, :rank]


def angles(x, y):
    cos = np.linalg.svd(x.T @ y, compute_uv=False).clip(0., 1.)
    return dict(principal_angles_deg=np.degrees(np.arccos(cos)).tolist(), subspace_overlap=float(np.square(cos).mean()))


def projection(x, basis):
    energy = np.square(x).sum(axis=-1)
    projected = np.square(x @ basis).sum(axis=-1)
    nonzero = energy > 1e-20
    ratios = projected[nonzero] / energy[nonzero]
    if not ratios.size:
        raise ValueError('No nonzero held-out gradient observations')
    return dict(aggregate_energy_fraction=float(projected.sum() / energy.sum()),
                mean_question_fraction=float(ratios.mean()),
                per_question_quantiles=np.quantile(ratios, [.1, .5, .9]).tolist(), n_nonzero=int(nonzero.sum()))


def learned_basis(checkpoint):
    obj = load(checkpoint)
    state = obj.get('state', obj.get('state_dict', obj))
    u = state['out.weight'].numpy().astype(np.float64)
    basis, s, _ = np.linalg.svd(u, full_matrices=False)
    keep = int((s > max(1e-12, s[0] * 1e-7)).sum())
    if keep == 0:
        raise ValueError('Frozen learned U has zero numerical rank')
    return basis[:, :keep], s.tolist()


def alignment(args, data, split):
    arc_path = args.out / 'arc_basis.pt'
    bases = load(arc_path)['spaces']
    result = {'arc_basis_sha256': sha(arc_path), 'gradient_projections': {}, 'angles_across_depths': {},
              'validation_or_external_only': split, 'held_out_centering': 'subtract mean of held-out vectors only for centered descriptive projection; training bases unchanged'}
    learned = None
    if args.checkpoint:
        learned, s = learned_basis(args.checkpoint)
        result['learned'] = {'sha256': sha(args.checkpoint), 'output_singular_values': s, 'numerical_rank': learned.shape[1]}
    for kind in ('shared', 'prefix'):
        domain = bases[kind]
        for k, values in data.items():
            x = values[kind].numpy().astype(np.float64)
            centered = x - x.mean(axis=0)
            if split == 'external':
                ext, ext_stats = decomposition(centered)
            for rank in RANKS:
                control = {'arc_depth_gradient_centered': domain[k]['centered'].numpy()[:, :rank],
                           'arc_depth_gradient_uncentered': domain[k]['uncentered'].numpy()[:, :rank],
                           'arc_joint_gradient_centered': domain['joint']['centered'].numpy()[:, :rank],
                           'arc_hidden_depth_pca': bases['hidden'][k]['centered'].numpy()[:, :rank],
                           'arc_hidden_joint_pca': bases['hidden']['joint']['centered'].numpy()[:, :rank],
                           'random': random_basis(rank)}
                if learned is not None:
                    control['learned_u'] = learned[:, :min(rank, learned.shape[1])]
                reference = control['arc_depth_gradient_centered']
                result['gradient_projections'][f'{kind}/{k}/r{rank}'] = {
                    method: dict(raw=projection(x, basis), centered=projection(centered, basis), rank=basis.shape[1],
                                 angle_to_arc_depth_gradient=angles(reference, basis)) for method, basis in control.items()}
                if split == 'external':
                    result['gradient_projections'][f'{kind}/{k}/r{rank}']['external_descriptive_spectrum'] = dict(
                        self_centered_energy_fraction=ext_stats['top_energy_fraction'][str(rank)],
                        angle_to_arc_depth_gradient=angles(reference, ext[:, :rank]))
        for i, a in enumerate(data):
            for b in list(data)[i + 1:]:
                for rank in RANKS:
                    result['angles_across_depths'][f'{kind}/{a}-{b}/r{rank}'] = angles(
                        domain[a]['centered'].numpy()[:, :rank], domain[b]['centered'].numpy()[:, :rank])
    dump(args.out / ('OPENBOOKQA_GRADIENT_ALIGNMENT.json' if split == 'external' else 'ARC_GRADIENT_ALIGNMENT.json'), result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('lock', 'fit', 'analyze', 'align', 'external', 'intervene'))
    for name in ('archive', 'corpus', 'model', 'out', 'external-data', 'freeze', 'checkpoint'):
        parser.add_argument(f'--{name}', type=Path, required=name == 'out')
    parser.add_argument('--train-n', type=int, default=320)
    parser.add_argument('--validation-n', type=int, default=299)
    parser.add_argument('--external-n', type=int, default=192)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--rank', type=int, default=8, choices=RANKS)
    parser.add_argument('--depths', type=int, choices=DEPTHS, nargs='+', default=DEPTHS)
    parser.add_argument('--dose', type=float, default=1.)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    if args.batch <= 0 or len(set(args.depths)) != len(args.depths):
        parser.error('Use a positive batch size and list each depth at most once')
    if args.phase in ('lock', 'fit', 'intervene') and not args.corpus:
        parser.error('This phase requires --corpus')
    if args.phase in ('fit', 'external', 'intervene') and (not args.archive or not args.model):
        parser.error('This phase requires --model and --archive')
    if args.phase == 'lock':
        cohorts(args)
    elif args.phase == 'fit':
        data, lock = cohorts(args)
        collect(args, data, lock)
        fit_analysis(args)
    elif args.phase == 'analyze':
        fit_analysis(args)
    elif args.phase == 'align':
        alignment(args, load_matrices(args, 'validation'), 'validation')
    elif args.phase == 'external':
        data, lock = cohorts(args, external=True)
        collect(args, data, lock)
        alignment(args, load_matrices(args, 'external'), 'external')
    elif args.phase == 'intervene':
        if not args.checkpoint:
            parser.error('Intervention requires the frozen --checkpoint')
        from subspace_intervene import intervene

        intervene(args)


if __name__ == '__main__':
    main()
