"""CPU audit after the frozen chain completes; never fits a replacement policy."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'frozen_loop_update_value_seed2021_v1'
ALPHAS = ['0.0', '-1.0', '-0.5', '0.5']
KINDS = ['input', 'confidence', 'combined']


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_rows(path):
    with path.open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_x(row, kind, action):
    d, c = row['decision'], row['controls']
    if 'losses' not in d:
        return None
    probe = [d['losses']['0.0'][part] - d['losses'][action][part] for part in ['A', 'B']]
    confidence = [c[action]['margin'] - c['0.0']['margin'], c['0.0']['entropy'] - c[action]['entropy']]
    return np.asarray(probe if kind == 'input' else confidence if kind == 'confidence' else probe + confidence)


def independent_choose(row, model):
    if 'losses' not in row['decision']:
        return '0.0', np.zeros(4)
    values = [0.]
    for action in ALPHAS[1:]:
        m = model['actions'][action]
        values.append(float(((independent_x(row, model['kind'], action) - m['mean']) / m['scale']) @ np.asarray(m['coef']) + m['intercept']))
    return ALPHAS[int(np.argmax(values))], np.asarray(values)


def main():
    state = load(ROOT / 'results/RUN_STATE.json')
    assert state['status'] == 'complete', 'Audit requires a terminal, complete chain'
    lock = load(ROOT / 'configs/LOCK.json')
    for rel, digest in lock['files'].items():
        assert sha(ROOT / rel) == digest, rel
    for rel, digest in lock['dependencies'].items():
        assert sha(ROOT.parent / rel) == digest, rel
    freeze = load(ROOT / 'configs/TRAIN_LOCK.json')
    selection = load(ROOT / 'configs/SELECTION_LOCK.json')
    models = load(ROOT / 'configs/heads.json')
    assert freeze['source_lock_sha256'] == sha(ROOT / 'configs/LOCK.json')
    assert freeze['heads_sha256'] == selection['heads_sha256'] == sha(ROOT / 'configs/heads.json')
    assert selection['train_lock_sha256'] == sha(ROOT / 'configs/TRAIN_LOCK.json')
    assert selection['analysis_sha256'] == sha(ROOT / 'results/VALIDATION_ANALYSIS.json')
    validation_exit = load(ROOT / 'results/validation_EXIT.json')
    assert freeze['frozen_unix'] < validation_exit['start_unix'], 'Head freeze must precede validation collection'
    assert all(run['exit_code'] == 0 for run in state['runs'])
    charged = sum(run['worker_seconds'] for run in state['runs'] if run['gpu_reserved'])
    budget = load(ROOT / 'results/BUDGET_RUNNING.json')
    assert abs(charged - budget['gpu_charge_seconds']) < 1e-6
    assert charged <= 10800 and state['end_unix'] <= state['start_unix'] + 10800

    splits = {}
    max_baseline_error = 0.
    for split in ['head_train', 'validation']:
        path = ROOT / 'results' / (split + '.jsonl')
        complete = load(ROOT / 'results' / (split + '_COMPLETE.json'))
        rows = read_rows(path)
        reference = {r['id']: r for r in read_rows(OLD / 'results' / (split + '.jsonl'))}
        assert len(rows) == complete['n'] == lock['manifests'][split]['n']
        assert [r['id'] for r in rows] == lock['manifests'][split]['ids']
        assert sha(path) == complete['sha256']
        assert complete['weights_unchanged'] and complete['all_nonzero_actions_propagated']
        assert complete['source_lock_sha256'] == sha(ROOT / 'configs/LOCK.json')
        for row in rows:
            den = np.asarray(row['choice_char_lengths'])
            assert (den > 0).all()
            for branch in [*row['actions'].values(), *row['terminal'].values()]:
                assert int(np.argmax(np.asarray(branch['sum_scores']) / den)) == branch['prediction']
            for drift in row['downstream_state_rms'].values():
                assert len(drift) == 4 and drift[0] == 0 and min(drift[1:]) > 0
            old = reference[row['id']]
            assert row['gold'] == old['gold']
            error = float(np.max(np.abs(np.asarray(row['actions']['0.0']['sum_scores']) - old['views']['official']['sum_scores']['adaptive'])))
            assert error < 1e-5
            max_baseline_error = max(max_baseline_error, error)
        splits[split] = rows
    assert not {r['id'] for r in splits['head_train']} & {r['id'] for r in splits['validation']}
    assert freeze['training_outcomes_sha256'] == sha(ROOT / 'results/head_train.jsonl')
    assert selection['validation_sha256'] == sha(ROOT / 'results/validation.jsonl')

    train = [r for r in splits['head_train'] if 'losses' in r['decision']]
    event_counts = {}
    max_coefficient_error = 0.
    for kind in KINDS:
        assert models[kind]['kind'] == kind and models[kind]['alpha'] == 100 and models[kind]['threshold'] == 0
        coefficients = 0
        for action in ALPHAS[1:]:
            m = models[kind]['actions'][action]
            X = np.stack([independent_x(r, kind, action) for r in train])
            y = np.asarray([int(r['actions'][action]['prediction'] == r['gold']) - int(r['actions']['0.0']['prediction'] == r['gold']) for r in train])
            mean, scale = X.mean(0), X.std(0)
            scale[scale < 1e-8] = 1
            np.testing.assert_allclose(mean, m['mean'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(scale, m['scale'], rtol=0, atol=1e-12)
            Z = (X - mean) / scale
            centered = Z - Z.mean(0)
            beta = np.linalg.solve(centered.T @ centered + 100 * np.eye(X.shape[1]), centered.T @ (y - y.mean()))
            intercept = float(y.mean() - Z.mean(0) @ beta)
            np.testing.assert_allclose(beta, m['coef'], rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(intercept, m['intercept'], rtol=1e-8, atol=1e-10)
            max_coefficient_error = max(max_coefficient_error, float(np.max(np.abs(beta - m['coef']))))
            np.testing.assert_allclose(np.quantile(Z @ beta + intercept, [.2, .4, .6, .8]), m['training_quintile_boundaries'], rtol=1e-8, atol=1e-10)
            coefficients += len(beta) + 1
            event_counts[action] = {'beneficial': int((y > 0).sum()), 'harmful': int((y < 0).sum()), 'neutral': int((y == 0).sum())}
            assert min(event_counts[action]['beneficial'], event_counts[action]['harmful']) >= 20
        assert coefficients == {'input': 9, 'confidence': 9, 'combined': 15}[kind]
        assert coefficients == models[kind]['linear_coefficients_including_intercepts']

    for split, filename in [('head_train', 'training_head_predictions.jsonl'), ('validation', 'validation_head_predictions.jsonl')]:
        stored = read_rows(ROOT / 'results' / filename)
        assert [r['id'] for r in stored] == [r['id'] for r in splits[split]]
        for row, record in zip(splits[split], stored):
            for kind in KINDS:
                action, values = independent_choose(row, models[kind])
                assert action == record['heads'][kind]['action']
                np.testing.assert_allclose(values, record['heads'][kind]['values'], rtol=0, atol=1e-12)
                if split == 'validation':
                    assert record['heads'][kind]['prediction'] == row['actions'][action]['prediction']
    report = load(ROOT / 'results/VALIDATION_ANALYSIS.json')
    validation = splits['validation']
    correct = np.asarray([[r['actions'][a]['prediction'] == r['gold'] for a in ALPHAS] for r in validation])
    chosen_actions = {kind: np.asarray([ALPHAS.index(independent_choose(r, models[kind])[0]) for r in validation]) for kind in KINDS}
    policies = {'adaptive': correct[:, 0], **{kind: correct[np.arange(len(validation)), indices] for kind, indices in chosen_actions.items()}, 'best_fixed_train': correct[:, ALPHAS.index(freeze['best_fixed_train'])]}
    raw_p = {}
    for kind in ['input', 'combined']:
        assert set(report['primary_comparisons'][kind]) == {'adaptive', 'confidence', 'best_fixed_train'}
        for comparator in ['adaptive', 'confidence', 'best_fixed_train']:
            delta = policies[kind].astype(int) - policies[comparator].astype(int)
            plus, minus, n = int((delta > 0).sum()), int((delta < 0).sum()), len(delta)
            s = report['primary_comparisons'][kind][comparator]
            assert (plus, minus, plus - minus) == (s['corrections'], s['degradations'], s['net'])
            assert int(policies[kind].sum()) == s['correct']
            p = float(binomtest(plus, plus + minus).pvalue) if plus + minus else 1.
            assert abs(p - s['mcnemar_p']) < 1e-12
            counts = np.random.default_rng(2021).multinomial(n, [minus / n, (n - minus - plus) / n, plus / n], size=10000)
            ci = np.quantile((counts[:, 2] - counts[:, 0]) / n * 100, [.025, .975])
            np.testing.assert_allclose(ci, s['ci95_pp'], rtol=0, atol=1e-12)
            raw_p[kind + '_vs_' + comparator] = p
        selected = chosen_actions[kind]
        available = np.flatnonzero(['losses' in r['decision'] for r in validation])
        rng = np.random.default_rng(2021)
        shuffled = []
        for _ in range(10000):
            action = selected.copy()
            action[available] = rng.permutation(action[available])
            shuffled.append(int(correct[np.arange(len(validation)), action].sum()))
        shuffle_p = float((1 + (np.asarray(shuffled) >= policies[kind].sum()).sum()) / 10001)
        assert abs(shuffle_p - report['shuffled'][kind]['p']) < 1e-12
        raw_p[kind + '_vs_shuffle'] = shuffle_p
    previous = 0.
    for i, key in enumerate(sorted(raw_p, key=raw_p.get)):
        previous = max(previous, min(1., (len(raw_p) - i) * raw_p[key]))
        assert abs(previous - report['primary_holm_eight_p'][key]) < 1e-12
    recomputed_eligible = []
    assert len(report['primary_holm_eight_p']) == 8
    for kind in ['input', 'combined']:
        admitted = all(s['net'] >= 5 and s['ci95_pp'][0] > 0 and report['primary_holm_eight_p'][kind + '_vs_' + comparator] < .05 for comparator, s in report['primary_comparisons'][kind].items())
        if admitted and report['primary_holm_eight_p'][kind + '_vs_shuffle'] < .05:
            recomputed_eligible.append(kind)
    assert recomputed_eligible == report['eligible'] == selection['eligible']
    expected = recomputed_eligible[0] if recomputed_eligible else 'adaptive'
    assert expected == report['selected_policy'] == selection['selected_policy']
    result = {'passed': True, 'scope': 'Independent CPU reconstruction of frozen source/data, least-squares heads, every stored decision and selection gate; no new evaluation or tuning', 'training_rows': len(splits['head_train']), 'validation_rows': len(splits['validation']), 'max_baseline_score_error': max_baseline_error, 'max_ridge_coefficient_error': max_coefficient_error, 'training_action_event_counts': event_counts, 'selected_policy': expected, 'gpu_charge_seconds': charged, 'source_lock_sha256': sha(ROOT / 'configs/LOCK.json'), 'analysis_sha256': sha(ROOT / 'results/VALIDATION_ANALYSIS.json')}
    (ROOT / 'results/INDEPENDENT_CPU_VERIFICATION.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
