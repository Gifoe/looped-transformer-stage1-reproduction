"""Project frozen feedback onto rank-matched spaces, then execute Ouro's native suffix."""

import json
import math
import time

import numpy as np
from scipy.stats import binomtest
import torch

from minimal import forward, load_net
from subspace import SEED, by_question, dump, load, random_basis, read, read_runtime, sha


class Projected:
    def __init__(self, net, basis, dose):
        self.net = net
        self.kind = net.kind
        self.basis = basis
        self.dose = dose
        self.totals = torch.zeros(7, dtype=torch.float32, device='cuda')
        self.max_error = torch.zeros(2, dtype=torch.float32, device='cuda')

    def delta(self, h, previous=None):
        original = self.net.delta(h, previous)
        base_energy = original.square().sum(dim=-1, keepdim=True)
        base_energy_norm = base_energy.sqrt()
        projected = original if self.basis is None else original @ self.basis @ self.basis.T
        proj_energy = projected.square().sum(dim=-1, keepdim=True)
        active = base_energy > 0
        missing = active & (proj_energy.sqrt() <= 1e-12 * base_energy_norm)
        if self.basis is not None:
            projected = torch.where(missing, self.basis[:, 0], projected)
        matched = projected * (base_energy_norm / projected.norm(dim=-1, keepdim=True).clamp_min(1e-20))
        if self.basis is None:
            matched = original
        effective = (h + self.dose * matched).to(torch.bfloat16).float() - h
        reference = (h + self.dose * original).to(torch.bfloat16).float() - h
        error_before = (matched.norm(dim=-1, keepdim=True) - base_energy_norm).abs() * abs(self.dose)
        error_after = (effective.norm(dim=-1, keepdim=True) - reference.norm(dim=-1, keepdim=True)).abs()
        self.max_error = torch.maximum(self.max_error, torch.stack((error_before.max(), error_after.max())))
        self.totals += torch.stack((base_energy.sum(), proj_energy.sum(), active.sum(), missing.sum(),
                                   reference.square().sum(), effective.square().sum(), error_after.sum()))
        return matched

    def diagnostics(self):
        full, projected, active, degenerate, effective_full, effective_edit, total_abs_difference = self.totals.tolist()
        before, after = self.max_error.tolist()
        return dict(unrescaled_projected_energy_fraction=projected / max(full, 1e-30), active_tokens=int(active),
                    degenerate_projection_tokens=int(degenerate), max_l2_norm_gap_before_bf16=before,
                    max_l2_norm_gap_after_bf16=after, quantized_energy_ratio=effective_edit / max(effective_full, 1e-30),
                    mean_absolute_quantized_l2_gap=total_abs_difference / max(active, 1.))


def comparative_stats(rows, names, depths):
    rng = np.random.default_rng(SEED)
    result = {}
    for k in depths:
        for metric in ('acc', 'acc_norm'):
            scores = {method: [np.array(r['scores'][f'{method}_k{k}']) / (r['characters'] if metric == 'acc_norm' else 1.)
                               for r in rows] for method in (*names, 'historic_native')}
            labels = [r['gold'] for r in rows]
            correct = {method: np.array([s.argmax() == g for s, g in zip(value, labels, strict=True)], dtype=int)
                       for method, value in scores.items()}
            margins = {method: np.array([s[g] - np.delete(s, g).max() for s, g in zip(value, labels, strict=True)])
                       for method, value in scores.items()}
            boot = rng.integers(0, len(rows), (12000, len(rows)))
            for method in names:
                for baseline in ('historic_native', 'full'):
                    if method == baseline:
                        continue
                    delta = correct[method] - correct[baseline]
                    margin = margins[method] - margins[baseline]
                    rescue, damage = int((delta > 0).sum()), int((delta < 0).sum())
                    p = float(binomtest(rescue, rescue + damage).pvalue) if rescue + damage else 1.
                    result[f'{method}/{baseline}/k{k}/{metric}'] = dict(
                        n=len(rows), correct=int(correct[method].sum()), baseline_correct=int(correct[baseline].sum()),
                        rescue=rescue, damage=damage, delta_pp=float(100 * delta.mean()),
                        delta_ci95_pp=(100 * np.quantile(delta[boot].mean(axis=1), [.025, .975])).tolist(),
                        mean_gold_margin_delta=float(margin.mean()),
                        mean_gold_margin_delta_ci95=np.quantile(margin[boot].mean(axis=1), [.025, .975]).tolist(),
                        p_exact_mcnemar=p)
    running = 0.
    ordered = sorted(result, key=lambda key: result[key]['p_exact_mcnemar'])
    for i, key in enumerate(ordered):
        running = max(running, min(1., (len(ordered) - i) * result[key]['p_exact_mcnemar']))
        result[key]['p_holm'] = running
    return result


@torch.no_grad()
def intervene(args):
    if not math.isfinite(args.dose) or args.dose == 0.:
        raise ValueError('Projection comparisons require a nonzero finite --dose')
    lock = args.out / 'ARC_GRADIENT_LOCK.json'
    basis_path = args.out / 'arc_basis.pt'
    basis = load(basis_path)
    if sha(lock) != basis['train_ids_sha256']:
        raise ValueError('Gradient basis targeting a different ARC cohort')
    manifest = read(lock)
    validation_data = args.corpus / 'validation.parquet'
    if sha(validation_data) != manifest['cohorts']['validation']['sha256']:
        raise ValueError('Validation source changed')
    old, model, tokenizer, runtime = read_runtime(args.archive, args.model)
    validation = old.read_data(args.corpus, 'validation')
    by_id = {r['id']: r for r in validation}
    rows = [by_id[x] for x in manifest['cohorts']['validation']['ids']]
    questions, info, order = by_question(old, tokenizer, rows)
    archived_scores_path = args.archive / 'results/validation_predictions.jsonl'
    with archived_scores_path.open(encoding='utf-8') as source:
        archived = {r['id']: r for r in map(json.loads, source)}
    for r in info:
        other = archived[r['id']]
        if any(r[key] != other[key] for key in ('gold', 'labels', 'characters', 'answer_tokens')):
            raise ValueError('Historical Native scoring protocol mismatch')
        r['scores'].update({f'historic_native_k{k}': other['scores'][f'native_k{k}'] for k in args.depths})
    net = load_net(args.checkpoint)
    rank = min(args.rank, net.rank)
    if net.kind == 'constant':
        rank = 1
    spaces = basis['spaces']
    cost = {}
    for k in args.depths:
        tag = f'k{k}'
        arms = {'full': None,
                'depth_gradient': spaces['shared'][tag]['centered'][:, :rank],
                'joint_gradient': spaces['shared']['joint']['centered'][:, :rank],
                'hidden_pca': spaces['hidden'][tag]['centered'][:, :rank],
                'random': torch.tensor(random_basis(rank), dtype=torch.float32)}
        arms = {name: Projected(net, x.cuda().contiguous() if x is not None else None, args.dose)
                for name, x in arms.items()}
        for name, arm in arms.items():
            elapsed = []
            padded = scored = 0
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for lo in range(0, len(order), args.batch):
                start = time.perf_counter()
                subset = [x for q in order[lo:lo + args.batch] for x in questions[q]]
                packed = old.pack(subset)
                final, _ = forward(model, packed, k, arm, dose=args.dose)
                scores = old.scores(model.lm_head(final).float(), packed[1].cpu().numpy(), packed[-1], len(subset))
                old.record(info, subset, {f'{name}_k{k}': scores})
                elapsed.append(time.perf_counter() - start)
                padded += packed[0].numel() * k
                scored += packed[-1].numel()
            diagnostics = arm.diagnostics()
            if diagnostics['degenerate_projection_tokens'] or diagnostics['max_l2_norm_gap_before_bf16'] > 1e-4:
                raise RuntimeError(('Invalid norm matching', name, k, diagnostics))
            cost[f'{name}/{tag}'] = dict(seconds=sum(elapsed), median_batch_ms=1000 * float(np.median(elapsed)),
                                         p95_batch_ms=1000 * float(np.quantile(elapsed, .95)), backbone_calls=len(elapsed),
                                         padded_token_loop_uses=padded, scored_tokens=scored, write_norm=diagnostics,
                                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                         peak_reserved_bytes=torch.cuda.max_memory_reserved())
            print('projection', k, name, cost[f'{name}/{tag}'], flush=True)
    file = args.out / 'SUBSPACE_INTERVENTION_SCORES.jsonl'
    temp = file.with_suffix('.tmp')
    old.write_predictions(temp, info)
    temp.replace(file)
    dump(args.out / 'SUBSPACE_INTERVENTION.json', dict(
        costs=cost, runtime=runtime, checkpoint_sha256=sha(args.checkpoint), historical_native_sha256=sha(archived_scores_path),
        operator_parameters=sum(p.numel() for p in net.parameters()), cohort_lock_sha256=sha(lock), basis_sha256=sha(basis_path),
        projected_rank=rank, learned_rank=net.rank, dose=args.dose,
        norm_protocol='Each token projection is rescaled to the original update L2 norm in FP32. BF16 runtime norm discrepancies are audited separately.',
        paired_results=comparative_stats(info, tuple(arms), args.depths), score_sha256=sha(file)))
