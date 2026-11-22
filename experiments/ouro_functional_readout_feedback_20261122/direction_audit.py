"""Audit the previous ARC readout with fixed-dose orientation and semantic controls."""

import argparse
from collections import Counter
from pathlib import Path
import json
import math
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.stats import binomtest
import torch


SEED = 261122991
DEPTHS = (1, 2, 3, 4)
CHANNELS = ('attention_first', 'attention_last', 'mlp_first', 'mlp_last')
MASKS = {
    'without_attention': CHANNELS[:2],
    'without_mlp': CHANNELS[2:],
    'without_attention_and_mlp': CHANNELS,
}
CONTROLS = ('gaussian_direction', 'cross_question', 'profile_matched', *MASKS)
METRICS = ('acc', 'acc_norm')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def write_json(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def write_jsonl(path, rows):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def check_rows(reference, other):
    for one, two in zip(reference, other, strict=True):
        for name in ('id', 'gold', 'labels', 'characters', 'answer_tokens'):
            if one[name] != two[name]:
                raise ValueError((one['id'], name, one[name], two[name]))


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def project(x, axis):
    return (x * axis).sum(dim=-1, keepdim=True) * axis


def geometry_control(h, w, true_shift, proposal, fallback):
    """Keep the update's projections on raw state and writes, and its norm."""
    write_axis = unit(w)
    raw_axis = unit(h - project(h, write_axis))
    fixed = project(true_shift, raw_axis) + project(true_shift, write_axis)
    residual = true_shift - fixed
    tangent = proposal - project(proposal, raw_axis) - project(proposal, write_axis)
    alternate = fallback - project(fallback, raw_axis) - project(fallback, write_axis)
    replaced = tangent.norm(dim=-1, keepdim=True) < 1e-8
    tangent = torch.where(replaced, alternate, tangent)
    return fixed + unit(tangent) * residual.norm(dim=-1, keepdim=True), int(replaced.sum())


def alignment(native, cache_rows, depth):
    """Generate donor identities from candidate metadata, without using gold."""
    columns = []
    ranks = []
    sizes = np.array([len(row['labels']) for row in cache_rows])
    for row, meta in zip(native, cache_rows, strict=True):
        s = np.asarray(row['scores'][f'native_k{depth}'])
        chars = np.asarray(meta['characters'])
        tokens = np.asarray(meta['answer_tokens'])
        order = np.argsort(-s, kind='stable')
        ranks.append(order)
        signal = np.stack((s, s / chars, s - s.max(), np.log(chars), np.log(tokens)))[:, order]
        # Four fixed raw-rank quantiles also define comparable profiles for 3/5-option items.
        columns.append(np.concatenate([np.interp(np.linspace(0, 1, 4), np.linspace(0, 1, len(s)), line)
                                       for line in signal]))
    profiles = np.stack(columns)
    scaled = (profiles - profiles.mean(axis=0)) / profiles.std(axis=0).clip(min=1e-8)
    distances = np.sqrt(((scaled[:, None] - scaled[None, :]) ** 2).mean(axis=-1))
    cost = distances + (sizes[:, None] != sizes[None, :]) * 100.
    np.fill_diagonal(cost, np.inf)
    _, matched = linear_sum_assignment(cost)
    rng = np.random.default_rng(SEED + depth)
    exchanged = np.empty_like(matched)
    for size in np.unique(sizes):
        members = rng.permutation(np.flatnonzero(sizes == size))
        if len(members) == 1:
            exchanged[members] = rng.choice(np.flatnonzero(sizes != size), size=1)
        else:
            exchanged[members] = np.roll(members, 1)
    result = {}
    for name, donors in (('cross_question', exchanged), ('profile_matched', matched)):
        if np.any(donors == np.arange(len(cache_rows))):
            raise ValueError('A question was assigned its own direction')
        by_question = []
        exact_chars = exact_tokens = total = 0
        raw_gaps = []
        norm_gaps = []
        for q, donor in enumerate(donors):
            rank_slots = np.rint(np.linspace(0, sizes[donor] - 1, sizes[q])).astype(int)
            pairing = np.empty(sizes[q], dtype=int)
            pairing[ranks[q]] = ranks[donor][rank_slots]
            by_question.append(pairing.tolist())
            for a, b in enumerate(pairing):
                c, d = cache_rows[q]['characters'][a], cache_rows[donor]['characters'][b]
                t, u = cache_rows[q]['answer_tokens'][a], cache_rows[donor]['answer_tokens'][b]
                x, y = native[q]['scores'][f'native_k{depth}'][a], native[donor]['scores'][f'native_k{depth}'][b]
                exact_chars += c == d
                exact_tokens += t == u
                raw_gaps.append(abs(x - y))
                norm_gaps.append(abs(x / c - y / d))
                total += 1
        result[name] = dict(donor_index=donors.tolist(), donor_options=by_question,
                            donor_ids=[cache_rows[i]['id'] for i in donors],
                            same_option_count=int((sizes == sizes[donors]).sum()),
                            candidate_pairs=total, exact_character_length_pairs=exact_chars,
                            exact_token_length_pairs=exact_tokens,
                            mean_absolute_raw_score_gap=float(np.mean(raw_gaps)),
                            mean_absolute_norm_score_gap=float(np.mean(norm_gaps)),
                            mean_standardized_profile_gap=float(distances[np.arange(len(donors)), donors].mean()))
    return result


def donor_tokens(dataset, donor_map, questions):
    indexes = []
    for q in questions:
        donor = donor_map['donor_index'][q]
        for a, (start, end) in enumerate(dataset.offsets[q]):
            src_start, src_end = dataset.offsets[donor][donor_map['donor_options'][q][a]]
            positions = np.rint(np.linspace(src_start, src_end - 1, end - start)).astype(int)
            indexes.extend(positions.tolist())
    return indexes


def masked_orientation(net, state, depth, names):
    """Remove learned input features and their RMS statistics, not the recipient writes."""
    masked = dict(state)
    for name in names:
        masked[name] = torch.zeros_like(state[name])
    result = []
    hook = net.align.register_forward_hook(lambda _, args, output: result.append(output))
    try:
        net(masked, depth)
    finally:
        hook.remove()
    return result[0]


def log_odds(scores):
    return [float(s - logsumexp(np.concatenate((scores[:i], scores[i + 1:]))))
            for i, s in enumerate(scores)]


def decisions(row):
    gold = row['gold']
    result = {}
    for name, values in row['scores'].items():
        raw = np.asarray(values)
        norm = raw / row['characters']
        result[name] = {}
        for metric, scores in zip(METRICS, (raw, norm), strict=True):
            odds = log_odds(scores)
            result[name][metric] = dict(winner=int(scores.argmax()), log_odds=odds,
                                        gold_margin=float(scores[gold] - np.delete(scores, gold).max()))
    return result


def paired(candidate, base):
    d = np.asarray(candidate, dtype=int) - np.asarray(base, dtype=int)
    rescued, damaged = int((d > 0).sum()), int((d < 0).sum())
    n = len(d)
    rng = np.random.default_rng(SEED)
    draws = rng.multinomial(n, np.array([damaged, n - rescued - damaged, rescued]) / n, size=30000)
    p = binomtest(rescued, rescued + damaged).pvalue if rescued + damaged else 1.
    return dict(n=n, correct=int(sum(candidate)), baseline_correct=int(sum(base)), rescued=rescued, damaged=damaged,
                accuracy=float(np.mean(candidate)), delta_pp=100 * float(d.mean()),
                ci95_pp=np.quantile(100 * (draws[:, 2] - draws[:, 0]) / n, [.025, .975]).tolist(),
                exact_mcnemar_p=float(p))


def summarize(rows, methods, baseline='native', control_reference=None):
    output = {}
    for reference in (baseline,) if control_reference is None else (baseline, control_reference):
        for method in methods:
            if method == reference:
                continue
            for k in DEPTHS:
                for metric in METRICS:
                    candidate = [r['audit'][f'{method}_k{k}'][metric] for r in rows]
                    base = [r['audit'][f'{reference}_k{k}'][metric] for r in rows]
                    labels = [r['gold'] for r in rows]
                    stats = paired([x['winner'] == g for x, g in zip(candidate, labels, strict=True)],
                                   [x['winner'] == g for x, g in zip(base, labels, strict=True)])
                    odds = np.array([x['log_odds'][g] - y['log_odds'][g]
                                     for x, y, g in zip(candidate, base, labels, strict=True)])
                    margin = np.array([x['gold_margin'] - y['gold_margin'] for x, y in zip(candidate, base, strict=True)])
                    output[f'{method}/{reference}/k{k}/{metric}'] = dict(stats, mean_gold_log_odds_delta=float(odds.mean()),
                                                                      mean_gold_margin_delta=float(margin.mean()))
    sorted_tests = sorted(output, key=lambda key: output[key]['exact_mcnemar_p'])
    previous = 0.
    for i, key in enumerate(sorted_tests):
        previous = max(previous, min(1., (len(sorted_tests) - i) * output[key]['exact_mcnemar_p']))
        output[key]['holm_p'] = previous
    return output


def fresh_audit(row, baseline='native'):
    row['audit'] = decisions(row)
    for name, result in row['audit'].items():
        reference = row['audit'][f'{baseline}_{name.split("_")[-1]}']
        for metric, observations in result.items():
            observations['log_odds_delta_vs_native'] = [a - b for a, b in zip(observations['log_odds'],
                                                                                reference[metric]['log_odds'], strict=True)]
            cached = row['audit'].get(f'native_cached_{name.split("_")[-1]}')
            if cached is not None:
                observations['log_odds_delta_vs_cached_native'] = [a - b for a, b in zip(
                    observations['log_odds'], cached[metric]['log_odds'], strict=True)]


def score(old, model, updated, target, group, count):
    logits = model.lm_head(model.model.norm(updated.to(torch.bfloat16))).float()
    return old.scores(logits, group, target, count).tolist()


@torch.inference_mode()
def validation(args, old, model, amplify, net, dose):
    data = amplify.Data(args.previous / 'features/validation')
    native_path = args.archive / 'results/validation_predictions.jsonl'
    frozen_path = args.previous / f'results/VALIDATION_{args.operator}.jsonl'
    native = read_jsonl(native_path)
    frozen = read_jsonl(frozen_path)
    check_rows(data.rows, native)
    check_rows(data.rows, frozen)
    if len(data.rows) != 299:
        raise ValueError('Expected all 299 ARC Validation questions')

    hashes = {}
    costs = {}
    geometry = {}
    maps = {}
    rows = [dict(r, scores={}) for r in data.rows]
    for path in (native_path, frozen_path, args.previous / 'features/validation/index.json'):
        hashes[path.name] = args.sha(path)
    for k in DEPTHS:
        begin = time.perf_counter()
        donor_maps = alignment(native, data.rows, k)
        maps[f'k{k}'] = donor_maps
        directions = []
        questions = list(range(len(rows)))
        for begin_batch in range(0, len(rows), args.batch):
            sub = questions[begin_batch:begin_batch + args.batch]
            state, *_ = data.batch(sub, k)
            _, (_, shift, _, _, _) = net(state, k, dose)
            directions.append(shift.cpu())
        direction_bank = torch.cat(directions)
        if len(direction_bank) != len(data.target):
            raise ValueError('Direction bank does not cover all answer tokens')
        del directions
        order_rng = torch.Generator(device='cuda').manual_seed(SEED + k)
        maximums = {name: np.zeros(3) for name in CONTROLS}
        degeneracies = Counter()
        torch.cuda.reset_peak_memory_stats()
        for begin_batch in range(0, len(rows), args.batch):
            sub = questions[begin_batch:begin_batch + args.batch]
            state, target, group, chars, counts, _ = data.batch(sub, k)
            index = group.cpu().numpy()
            edited, (gain, shift, wr, w, _) = net(state, k, dose)
            h = state['raw'][:, k - 1].float()
            native_write = gain * w
            update = dose * (native_write + shift)
            versions = {
                'native_cached': score(old, model, h, target, index, len(chars)),
                'full': score(old, model, edited, target, index, len(chars)),
                'magnitude_only': score(old, model, h + dose * native_write, target, index, len(chars)),
            }
            random = torch.randn(shift.shape, device='cuda', generator=order_rng)
            for name in CONTROLS:
                if name in donor_maps:
                    proposal = direction_bank[donor_tokens(data, donor_maps[name], sub)].cuda()
                elif name in MASKS:
                    proposal = masked_orientation(net, state, k, MASKS[name])
                else:
                    proposal = random
                changed, replaced = geometry_control(h, w, shift, proposal, random)
                degeneracies[name] += replaced
                altered = dose * (native_write + changed)
                residuals = torch.stack(((amplify.rms(update) - amplify.rms(altered)).abs().max(),
                                         (amplify.rms(h + update) - amplify.rms(h + altered)).abs().max(),
                                         (((update - altered) * w).mean(-1).abs() / wr[:, 0]).max()))
                maximums[name] = np.maximum(maximums[name], residuals.cpu().numpy())
                versions[name] = score(old, model, h + altered, target, index, len(chars))
            offset = 0
            for q, count in zip(sub, counts, strict=True):
                r = rows[q]['scores']
                r[f'native_k{k}'] = native[q]['scores'][f'native_k{k}']
                r[f'archive_full_k{k}'] = frozen[q]['scores'][f'k{k}']
                for name, values in versions.items():
                    r[f'{name}_k{k}'] = values[offset:offset + count]
                offset += count
        if any(v.max() >= .002 for v in maximums.values()):
            raise ValueError(('Dose geometry did not match', k, maximums))
        torch.cuda.synchronize()
        costs[f'k{k}'] = dict(wall_s=time.perf_counter() - begin, cached_questions=len(rows),
                              candidate_tokens=len(data.target), candidate_count=data.candidates, backbone_calls=0,
                              lm_head_calls=math.ceil(len(rows) / args.batch) * (3 + len(CONTROLS)),
                              peak_allocated_bytes=torch.cuda.max_memory_allocated())
        geometry[f'k{k}'] = dict(max_geometry_residuals={name: v.tolist() for name, v in maximums.items()},
                                 fallback_token_counts=dict(degeneracies))
        print('direction validation', k, costs[f'k{k}'], flush=True)
    parity = {}
    for row in rows:
        fresh_audit(row)
    for k in DEPTHS:
        for actual, expected in (('full', 'archive_full'), ('native_cached', 'native')):
            a = [v for r in rows for v in r['scores'][f'{actual}_k{k}']]
            b = [v for r in rows for v in r['scores'][f'{expected}_k{k}']]
            diff = abs(np.asarray(a) - b)
            parity[f'{actual}/{expected}/k{k}'] = dict(max_abs_score_delta=float(diff.max()),
                                                      mean_abs_score_delta=float(diff.mean()))
    write_jsonl(args.out / 'DIRECTION_VALIDATION_SCORES.jsonl', rows)
    write_json(args.out / 'DIRECTION_DONORS.json', maps)
    audit = summarize(rows, ('full', 'magnitude_only', *CONTROLS), 'native_cached', 'full')
    # Cached native is the controlled head-only comparator. The published native is kept for parity.
    return dict(results=audit, archive_parity=parity, geometry=geometry, cost=costs, input_sha256=hashes)


def archive_test(args):
    native_path = args.archive / 'results/test_predictions.jsonl'
    real_path = args.previous / 'results/TEST_PREDICTIONS.jsonl'
    rotated_path = args.previous / 'results/GEOMETRY_PREDICTIONS.jsonl'
    native, real, rotated = map(read_jsonl, (native_path, real_path, rotated_path))
    check_rows(native, real)
    check_rows(native, rotated)
    if len(native) != 1172:
        raise ValueError('Expected the complete published ARC Test split')
    rows = []
    for n, v, w in zip(native, real, rotated, strict=True):
        row = dict(n, scores={})
        for k in DEPTHS:
            row['scores'].update({f'native_k{k}': n['scores'][f'native_k{k}'],
                                  f'full_k{k}': v['scores'][f'{args.operator}_k{k}'],
                                  f'archived_rotation_k{k}': w['scores'][f'{args.operator}_rotated_k{k}']})
        fresh_audit(row)
        rows.append(row)
    return dict(results=summarize(rows, ('full', 'archived_rotation'), control_reference='full'),
                source='Frozen historical orientation scores, read from archive. No ARC Test model calls.',
                input_sha256={p.name: args.sha(p) for p in (native_path, real_path, rotated_path)})


def historic_quartets(args):
    source = args.csc / 'NATIVE.json'
    operators = args.csc / 'OPERATOR_ABLATION_RAW.json'
    inputs = read_json(source)['rows']
    heldout = {r['id']: r for r in inputs if r['split'] in ('test', 'shift')}
    groups = {}
    for form in ('borrow', 'carry'):
        for split in ('test', 'shift'):
            rows = [row for row in heldout.values() if (row['form'], row['split']) == (form, split)]
            outcomes = {}
            for k in (1, 2):
                free = [r['native'][k - 1] for r in rows]
                gold = [[b - a if form == 'borrow' else a + b for a, b in r['corners']] + [r['target']] for r in rows]
                exact = [[x['number'] == y for x, y in zip(guess, target, strict=True)]
                         for guess, target in zip(free, gold, strict=True)]
                outcomes[f'k{k}'] = dict(corners_correct=np.asarray(exact).sum(0).tolist(),
                                         all_corners_correct=sum(all(row[:4]) for row in exact),
                                         joint_target_prediction=sum(guess[3]['number'] == r['target'] for r, guess in zip(rows, free, strict=True)),
                                         joint_foil_prediction=sum(guess[3]['number'] == r['foil'] for r, guess in zip(rows, free, strict=True)))
            groups[f'{form}/{split}'] = dict(n=len(rows), outcomes=outcomes)
    archived_operators = {}
    for depth, rows in read_json(operators).items():
        for kind in ('joint', 'safe'):
            r = [row for row in rows if row['kind'] == kind and row['id'] in heldout]
            if r:
                archived_operators[f'{depth}/{kind}'] = {
                    name: dict(n=len(r), correct=sum(row['pred'][name]['number'] == row['target'] for row in r))
                    for name in r[0]['pred']}
    return dict(source='Historical free-generation controlled arithmetic with different trained operators.', native=groups,
                operators=archived_operators, input_sha256={p.name: args.sha(p) for p in (source, operators)})


def semantic_deck(path):
    deck = read_json(path)
    variants = ('base', 'paraphrase', 'change', 'hidden_permutation', 'visible_base', 'visible_permutation')
    rows = []
    metadata = []
    for family in deck:
        original = family['base_gold']
        changed = 1 - original
        if len(family['options']) != 2 or original not in (0, 1):
            raise ValueError(family)
        for v in variants:
            permutation = [1, 0] if v.endswith('permutation') else [0, 1]
            options = [family['options'][i] for i in permutation]
            question = family.get(v, family['base'])
            value = changed if v == 'change' else original
            labels = ['A', 'B']
            if v.startswith('visible_'):
                question += '\nOptions: ' + ' '.join(f'{label}. {option}' for label, option in zip(labels, options, strict=True))
            identifier = f'{family["id"]}/{v}'
            rows.append(dict(id=identifier, question=question, choices={'label': labels, 'text': options},
                             answerKey=labels[permutation.index(value)]))
            metadata.append(dict(family=family['id'], variant=v, canonical_options=permutation,
                                 options=options, question=question, rationale=family['rationale']))
    return deck, rows, metadata


def semantic_summary(deck, rows):
    out = {}
    indexed = {(r['family'], r['variant']): r for r in rows}
    pairs = dict(paraphrase=('base', 'paraphrase'), semantic_change=('base', 'change'),
                 hidden_permutation=('base', 'hidden_permutation'), visible_permutation=('visible_base', 'visible_permutation'))
    for k in DEPTHS:
        for metric in METRICS:
            for name, variants in pairs.items():
                result = {}
                for method in ('native', 'full'):
                    key = f'{method}_k{k}'
                    predictions = Counter()
                    odds = []
                    intervention = []
                    per_family = []
                    for family in deck:
                        a, b = (indexed[(family['id'], v)] for v in variants)
                        az, bz = (r['audit'][key][metric] for r in (a, b))
                        x, y = [r['canonical_options'][z['winner']] for r, z in ((a, az), (b, bz))]
                        ix = family['base_gold']
                        a_idx, b_idx = (r['canonical_options'].index(ix) for r in (a, b))
                        delta = bz['log_odds'][b_idx] - az['log_odds'][a_idx]
                        modifier = bz['log_odds_delta_vs_native'][b_idx] - az['log_odds_delta_vs_native'][a_idx]
                        odds.append(delta)
                        intervention.append(modifier)
                        both = az['winner'] == a['gold'] and bz['winner'] == b['gold']
                        predictions['both_correct'] += both
                        predictions['same_canonical_choice'] += x == y
                        predictions['different_canonical_choice'] += x != y
                        predictions['moves_old_gold_log_odds_down'] += delta < -1e-6
                        per_family.append(dict(id=family['id'], both_correct=both, same_choice=x == y,
                                               old_gold_log_odds_delta=delta, intervention_contrast=modifier))
                    result[method] = dict(n=len(deck), **dict(predictions),
                                          median_absolute_log_odds_response=float(np.median(np.abs(odds))),
                                          mean_old_gold_log_odds_change=float(np.mean(odds)),
                                          mean_intervention_contrast=float(np.mean(intervention)), pairs=per_family)
                result['full_vs_native_both_correct'] = paired(
                    [p['both_correct'] for p in result['full']['pairs']],
                    [p['both_correct'] for p in result['native']['pairs']])
                out[f'{name}/k{k}/{metric}'] = result
    return out


@torch.inference_mode()
def semantics(args, old, model, tok, features, net, dose):
    begin = time.perf_counter()
    deck, rows, metadata = semantic_deck(args.deck)
    items, info = old.encode(tok, rows)
    items.sort(key=lambda x: len(x[0]))
    for r, meta in zip(info, metadata, strict=True):
        r.update(meta)
    calls = 0
    torch.cuda.reset_peak_memory_stats()
    for i in range(0, len(items), args.batch):
        sub = items[i:i + args.batch]
        state, group, target = features.forward(model, old, sub, max(DEPTHS))
        calls += 1
        ix = group.cpu().numpy()
        for k in DEPTHS:
            h = state['raw'][:, k - 1]
            updated, _ = net(state, k, dose)
            old.record(info, sub, {f'native_k{k}': score(old, model, h, target, ix, len(sub)),
                                   f'full_k{k}': score(old, model, updated, target, ix, len(sub))})
    for r in info:
        fresh_audit(r)
    torch.cuda.synchronize()
    cost = dict(wall_s=time.perf_counter() - begin, model_calls=calls, questions=len(rows), candidates=len(items),
                lm_head_calls=calls * len(DEPTHS) * 2, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write_jsonl(args.out / 'SEMANTIC_SENSITIVITY_SCORES.jsonl', info)
    return dict(results=semantic_summary(deck, info), cost=cost, deck_sha256=args.sha(args.deck))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'archive', 'csc', 'model', 'out'):
        ap.add_argument(name, type=Path)
    ap.add_argument('--deck', type=Path, default=Path(__file__).with_name('semantic_audit_deck.json'))
    ap.add_argument('--operator', choices=('completion', 'direction'), default='completion')
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--phase', choices=('validation', 'semantics', 'archive', 'all'), default='all')
    args = ap.parse_args()
    if args.batch < 1:
        ap.error('--batch must be positive')
    sys.path.insert(0, str(args.previous.resolve()))
    import amplify
    import arc_features

    args.sha = arc_features.sha
    args.out.mkdir(parents=True, exist_ok=True)
    locked = read_json(args.previous / 'results/FREEZE.json')
    selected = locked['chosen'][args.operator]
    ckpt = args.previous / 'results' / selected['file']
    if locked['hashes'] != amplify.digest_bundle(args.previous / 'results', locked['chosen']):
        raise ValueError('Previous readout bundle failed checksum verification')
    output = dict(direction_operator=args.operator, frozen_dose=selected['dose'], checkpoint_sha256=args.sha(ckpt),
                  implementation_sha256=args.sha(Path(__file__)), seed=SEED)
    if args.phase in ('archive', 'all'):
        output['archive_test'] = archive_test(args)
        output['archive_arithmetic'] = historic_quartets(args)
        write_json(args.out / 'DIRECTION_ARCHIVE_AUDIT.json', output)
    if args.phase != 'archive':
        old = arc_features.archive(args.archive)
        model, tok, runtime = old.load_runtime(args.model)
        model.requires_grad_(False)
        net = amplify.operator(ckpt)
        output['runtime'] = runtime
        output['parameters'] = sum(p.numel() for p in net.parameters())
        meta = {name: output[name] for name in ('direction_operator', 'frozen_dose', 'checkpoint_sha256',
                                               'implementation_sha256', 'seed', 'runtime', 'parameters')}
        if args.phase in ('validation', 'all'):
            result = validation(args, old, model, amplify, net, selected['dose'])
            write_json(args.out / 'DIRECTION_VALIDATION_AUDIT.json', dict(meta, validation=result))
        if args.phase in ('semantics', 'all'):
            semantic = semantics(args, old, model, tok, arc_features, net, selected['dose'])
            write_json(args.out / 'SEMANTIC_SENSITIVITY_AUDIT.json', dict(meta, semantics=semantic))


if __name__ == '__main__':
    main()
