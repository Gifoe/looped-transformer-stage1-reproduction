"""Complete128 metric appendix from saved endpoints, independent of producer code.

Adds the global-K1 wrong-competitor metric and per-group reduced-base accounting.
Runs only after all five full-stage numerical audits pass. No model inference.
"""
import hashlib
import json
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction(values):
    return max(range(len(values)), key=lambda i: values[i])


def metrics(values, gold, fixed, global_fixed):
    wrong = [values[i] for i in range(len(values)) if i != gold]
    return {'meanwrong': values[gold] - sum(wrong) / len(wrong),
            'fixedwrong': values[gold] - values[fixed],
            'dynamicmargin': values[gold] - max(wrong),
            'global_K1_fixedwrong': values[gold] - values[global_fixed]}


def main():
    cohort_path = ROOT / 'inputs/confirmation128/cohort.json'
    labels_path = ROOT / 'inputs/confirmation128/analysis_labels.json'
    expected = set(read(cohort_path)['ids'])
    assert len(expected) == 128
    labels = read(labels_path)
    buckets, rows, provenance = defaultdict(list), [], []
    native_counts = {str(d): 0 for d in range(1, 5)}
    for stage in ['base', 'query', 'r2', 'r3', 'r4']:
        audit_path = ROOT / 'results/audits' / f'CONFIRMATION_{stage}_full128_v1_AUDIT.json'
        audit = read(audit_path)
        assert audit['status'] == 'PASS_COMPLETE' and not audit['failures']
        assert audit['n_observed'] == audit['n_expected'] == 128
        directory = ROOT / 'results' / f'confirmation_{stage}_full128_v1'
        seen = set()
        for result_path in sorted(directory.glob('*.results.json')):
            obj = read(result_path)
            native_path = result_path.with_name(result_path.name.replace('.results.json', '.native.json'))
            native = read(native_path)
            ident, split = obj['id'], obj['split']
            assert ident not in seen and ident in expected
            seen.add(ident)
            assert sha(native_path) == obj['native_file_sha256']
            gold = labels[split + ':' + ident]['gold']
            first = native['native']['1'] if stage in ['base', 'query'] else native['reused_K1_native']
            if stage == 'base':
                for depth in range(1, 5):
                    native_counts[str(depth)] += prediction(native['native'][str(depth)]['candidate_logits']) == gold
            for endpoint in obj['rows']:
                depth = endpoint['receiver_loop']
                baseline = native['native'][str(depth)]
                previous = native['native'][str(max(1, depth - 1))] if stage == 'base' else (
                    first if stage == 'query' else native['reused_donor_native'])
                fixed = max((i for i in range(len(baseline['scores'])) if i != gold), key=lambda i: baseline['scores'][i])
                global_fixed = max((i for i in range(len(first['scores'])) if i != gold), key=lambda i: first['scores'][i])
                bc = prediction(baseline['candidate_logits']) == gold
                pc = prediction(previous['candidate_logits']) == gold
                group = ('stable_correct' if pc else 'rescue') if bc else ('damage' if pc else 'stable_wrong')
                ec = prediction(endpoint['candidate_logits']) == gold
                exact_b = metrics(list(map(Fraction, baseline['candidate_logits'])), gold, fixed, global_fixed)
                exact_e = metrics(list(map(Fraction, endpoint['candidate_logits'])), gold, fixed, global_fixed)
                bm, em = [metrics(e['scores'], gold, fixed, global_fixed) for e in [baseline, endpoint]]
                record = {'stage': stage, 'id': ident, 'split': split, 'receiver_loop': depth,
                          'condition': endpoint['condition'], 'group': group,
                          'correct': int(ec), 'native_correct': int(bc),
                          'rescue': int(not bc and ec), 'damage': int(bc and not ec),
                          'top_tie': int(endpoint['candidate_logits'].count(max(endpoint['candidate_logits'])) > 1),
                          'fixed_wrong_index': fixed, 'global_K1_fixed_wrong_index': global_fixed,
                          'metrics': {key: {'value': float(em[key]), 'delta': float(em[key] - bm[key]),
                                           'exact_delta': float(exact_e[key] - exact_b[key]),
                                           'exact_sign': int(exact_e[key] > exact_b[key]) - int(exact_e[key] < exact_b[key])}
                                      for key in em}}
                rows.append(record)
                for stratum in ['all', group]:
                    buckets[(stage, depth, endpoint['condition'], stratum)].append(record)
            provenance.append({'stage': stage, 'id': ident, 'native_sha256': sha(native_path),
                               'result_sha256': sha(result_path)})
        assert seen == expected
    summary = []
    for (stage, depth, condition, stratum), observations in sorted(buckets.items()):
        result = {'stage': stage, 'receiver_loop': depth, 'condition': condition,
                  'stratum': stratum, 'n': len(observations)}
        for key in ['correct', 'native_correct', 'rescue', 'damage', 'top_tie']:
            result[key] = sum(row[key] for row in observations)
        result['metrics'] = {}
        for metric in observations[0]['metrics']:
            values = [row['metrics'][metric] for row in observations]
            result['metrics'][metric] = {
                'mean_absolute': statistics.mean(v['value'] for v in values),
                'mean_delta': statistics.mean(v['delta'] for v in values),
                'median_delta': statistics.median(v['delta'] for v in values),
                **{name: sum(v['exact_sign'] == sign for v in values)
                   for name, sign in [('exact_positive', 1), ('exact_negative', -1), ('exact_zero', 0)]}}
        summary.append(result)
    out = ROOT / 'results/ROOT_CONFIRMATION128_ALL_METRICS.json'
    out.write_text(json.dumps({'scope': 'All saved128 endpoints, four metrics, all/adjacent natural groups. '
                              'FP32 diagnostic conditions have only the fixed8 questions; their rows are not pooled '
                              'with production endpoints. Native K1 self-grouping has no adjacent correction claim.',
                              'n': 128, 'native_correct': native_counts,
                              'script_sha256': sha(Path(__file__)), 'cohort_sha256': sha(cohort_path),
                              'analysis_labels_sha256': sha(labels_path), 'summary': summary,
                              'source_files': provenance, 'rows': rows}, indent=2), encoding='utf8')
    print(json.dumps({'output': str(out), 'question_condition_rows': len(rows),
                      'summary_rows': len(summary), 'native_correct': native_counts, 'sha256': sha(out)}))


if __name__ == '__main__':
    main()
