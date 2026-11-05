"""CPU postrun audit: rerun the frozen integrity gate, independently recompute
the saved full-V representative projection and all already-unsealed pairing
statistics. No model, new inference, heldout input or second method selection.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads((ROOT / p).read_text(encoding='utf-8'))


def independent_clock_relations():
    """Read actual arrays across old-native/new-clock archives, before effects."""
    from r30_protocol import build
    _, tasks, _, plan = build()
    current = read('results/r30_EXECUTION_AUDIT.json')
    previous = read('results/r28_EXECUTION_AUDIT.json')
    natives = {x['task']: x['native'] for x in previous['checks']
               if x['type'] == 'no_hook_capture_exact'}
    captures = {x['task']: x for x in current['checks']
                if x['type'] == 'capture_only_native_exact'}
    assert set(captures) == set(plan['representative_native'].values())
    count = dict(old_native_full_states=0, old_native_heads=0,
                 CF_full_first3_states=0, capture_to_CF_loop1_donors=0,
                 identity_full_states=0, identity_heads=0,
                 repeat_full_states=0, repeat_heads=0, repeat_donors=0)
    with np.load(ROOT/'results/r30_REPRESENTATIVE_VECTORS.npz', allow_pickle=False) as now, \
         np.load(ROOT/'results/r28_REPRESENTATIVE_VECTORS.npz', allow_pickle=False) as old:
        def equal(a, b, archive_a=now, archive_b=now):
            assert a['dtype'] == b['dtype'] and a['shape'] == b['shape']
            x, y = archive_a[a['key']], archive_b[b['key']]
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        def snapshot(a, b, kind):
            assert len(a['states']) == len(b['states']) == 4
            for x, y in zip(a['states'], b['states']):
                equal(x, y); count[kind+'_full_states'] += 1
            assert set(a['heads']) == set(b['heads']) == {'4'}
            equal(a['heads']['4'], b['heads']['4']); count[kind+'_heads'] += 1
        for task, item in captures.items():
            assert item['exact'] and len(item['raw']['states']) == 4
            for a, b in zip(item['raw']['states'], natives[task]['states']):
                equal(a, b, now, old); count['old_native_full_states'] += 1
            equal(item['raw']['heads']['4'], natives[task]['heads']['4'], now, old)
            count['old_native_heads'] += 1
        for item in current['checks']:
            kind = item['type']
            if kind == 'CF_first3_full_and_loop1_source_exact':
                source = captures[tasks[item['task']]['native_task']]
                assert item['native'] == source['raw'] and len(item['donors']) == len(source['donors']) == 24
                for a, b in zip(item['native']['states'][:3], item['CF']['states'][:3]):
                    equal(a, b); count['CF_full_first3_states'] += 1
                for a, b in zip(item['donors'], source['donors']):
                    equal(a, b); count['capture_to_CF_loop1_donors'] += 1
            elif kind == 'identity_exact':
                assert item['native'] == captures[item['task']]['raw']
                snapshot(item['native'], item['identity'], 'identity')
            elif kind == 'CF_repeat_exact':
                snapshot(item['first'], item['repeat'], 'repeat')
                first = next(x for x in current['checks']
                             if x['type'] == 'CF_first3_full_and_loop1_source_exact'
                             and x['task'] == item['task'])
                assert len(item['donors']) == len(first['donors']) == 24
                for a, b in zip(item['donors'], first['donors']):
                    equal(a, b); count['repeat_donors'] += 1
    assert count == dict(old_native_full_states=8, old_native_heads=2,
                         CF_full_first3_states=6, capture_to_CF_loop1_donors=48,
                         identity_full_states=8, identity_heads=2,
                         repeat_full_states=8, repeat_heads=2, repeat_donors=48), count
    return dict(status='passed_actual_array_byte_comparisons', counts=count,
                limitation='Full-sequence and source-value comparisons cover the two frozen representatives, not every prefix.')


def main():
    from r30_analyze import numeric_integrity
    from r30_common import LOCK, gate

    spec = gate()
    complete, scores, rows, unmatched = numeric_integrity(spec)
    analysis = read('results/r30_ANALYSIS.json')
    assert analysis['protocol_sha256'] == hashlib.sha256((ROOT / LOCK).read_bytes()).hexdigest()
    cross_file = independent_clock_relations()
    book = read('logs/r30_gpu_budget.json')
    output = dict(status='frozen_numeric_integrity_gate_rerun_passed',
                  independent_scope='Representative full-V projection and already-unsealed score/pairing arithmetic; the frozen numeric_integrity implementation was rerun rather than claimed as a second independent implementation.',
                  protocol_sha256=analysis['protocol_sha256'],
                  actual_worker_seconds=book['used_seconds'],
                  total_gpu_worker_seconds=book['total_used_seconds'],
                  remaining_seconds=21600-book['total_used_seconds'],
                  model_run=False, new_GPU_authorized=False,
                  independent_cross_file_clock_relations=cross_file)
    if unmatched:
        assert analysis['status'] == 'complete_but_unmatched_no_effect_unsealing'
        output.update(status='numeric_complete_controls_unmatched_no_effect_audit',
                      unmatched=unmatched, effects_unsealed_by_Root=False)
    else:
        assert analysis['status'] == 'complete_analyzed_once'
        audit = read('results/r30_EXECUTION_AUDIT.json')
        first = next(r for r in audit['checks'] if r['type'] == 'first_real_math')
        with np.load(ROOT / 'results/r30_REPRESENTATIVE_VECTORS.npz', allow_pickle=False) as archive:
            arrays = {k: archive[first[k]['key']].copy() for k in ('q1', 'q4', 'n1', 'n4', 'b4')}
        q1, q4 = arrays['q1'].reshape(-1), arrays['q4'].reshape(-1)
        n4, b4 = arrays['n4'], arrays['b4']
        assert all(a.dtype == np.float32 and np.isfinite(a).all() for a in arrays.values())
        g = np.float32(.5) * (q4-q1)
        e = np.float32(.5) * (n4[1:]-b4)
        p = np.exp(q4.astype(np.float64)-float(q4.max()))
        p /= p.sum()

        def center(x):
            s = x.astype(np.float64)-x[..., :1].astype(np.float64)
            return s-(p*s).sum(axis=-1, keepdims=True)

        gc, ec = center(g), center(e)
        if np.array_equal(ec, np.broadcast_to(ec[:1], ec.shape)):
            ec = ec[:1]
        mean = ec.mean(axis=0)
        denominator = float(np.mean(np.sum(p*ec*ec, axis=-1)))
        numerator = float(np.sum(p*gc*mean))
        threshold = float(64*np.finfo(np.float64).eps*np.mean(np.max(np.abs(ec), axis=-1)**2))
        alpha = 0. if denominator <= threshold else float(np.clip(numerator/denominator, 0., 1.))
        h = alpha*mean
        main2 = float(np.sum(p*h*h))
        g2 = float(np.sum(p*gc*gc))
        r2 = float(np.sum(p*(gc-h)**2))
        rho = math.sqrt(max(r2/g2, 0.)) if g2 > 0 else 1.
        first_q = first['task']
        observed = scores['diagnostics'][first_q]
        recomputed = dict(alpha=alpha, numerator=numerator, denominator=denominator,
                          indistinguishable_threshold=threshold, main_norm2=main2,
                          scale_g2=g2, scale_r2=r2, scale_rho=rho)
        # The CPU agreement tolerance was fixed before the job; no gate is tuned from effects.
        errors = {k: abs(v-observed[k]) for k, v in recomputed.items()}
        assert all(v <= 1e-10 for v in errors.values()), errors
        assert ((denominator <= threshold) == observed['indistinguishable'])
        unsealed = {r['id']: r for r in analysis['predictions']}
        assert set(unsealed) == {r['id'] for r in rows} and len(unsealed) == 24
        methods = list(scores['rows'][0]['candidates'][0]['scores']) + ['R6_historical', 'Fixed_historical']
        correct = {m: [] for m in methods}
        predictions = []
        for record in scores['rows']:
            supplied = unsealed[record['id']]
            gold = supplied['gold']
            assert isinstance(gold, int) and 0 <= gold < len(record['candidates'])
            preds = {}
            for m in methods:
                if m.endswith('_historical'):
                    pred = supplied['predictions'][m]
                else:
                    values = np.asarray([c['scores'][m] for c in record['candidates']], dtype=np.float64)
                    assert np.isfinite(values).all()
                    pred = int(np.argmax(values))
                    assert pred == supplied['predictions'][m]
                preds[m] = pred
                correct[m].append(pred == gold)
            predictions.append(dict(id=record['id'], gold=gold, predictions=preds))
        counts = {m: int(sum(v)) for m, v in correct.items()}
        assert counts == analysis['counts']

        def pair(m, b):
            rescues = [i for i, (a, c) in enumerate(zip(correct[m], correct[b])) if a and not c]
            damages = [i for i, (a, c) in enumerate(zip(correct[m], correct[b])) if c and not a]
            return dict(rescue=len(rescues), damage=len(damages), net=len(rescues)-len(damages))

        retention, pairs = {}, {}
        for base, fixed in [('R6_historical', 'Fixed_historical'), ('R6_matched', 'Fixed')]:
            ids = [i for i, (a, b) in enumerate(zip(correct[base], correct[fixed])) if a and not b]
            retention[base] = {}
            for m in ('primary', 'LOO_N1', 'LOO_N2'):
                retention[base][m] = dict(total=len(ids), retained=sum(correct[m][i] for i in ids),
                                           ids=[rows[i]['id'] for i in ids])
            pairs[base] = {m: pair(m, base) for m in methods}
        assert pairs == analysis['pairs'] and retention == analysis['retention']
        protected = all(retention[b]['primary']['retained'] == retention[b]['primary']['total'] for b in retention)
        cost = protected and all(pairs[b]['primary']['net'] >= 0 for b in pairs)
        strong = (protected and all(pairs[b]['primary']['rescue'] >= 2 and pairs[b]['primary']['damage'] == 0 for b in pairs)
                  and all(counts['primary'] > counts[m] for m in ('M3', 'J2_guidance', 'G_matched', 'vocabulary_null', 'matched_scale'))
                  and all(pairs[b][m]['net'] >= 0 and retention[b][m]['retained'] == retention[b][m]['total']
                          for b in pairs for m in ('LOO_N1', 'LOO_N2')))
        assert strong == analysis['strong_screen'] and cost == analysis['cost_review_eligible']
        diagnostics = scores['diagnostics']
        graph = read('configs/r28_TRUE_PREFIX_TASK_INVENTORY.json')
        active_unique = sum(d['alpha'] > 0 for d in diagnostics.values())
        active_uses = sum(diagnostics[u['tasks']['Q']]['alpha'] > 0 for u in graph['usages'])
        output.update(status='passed_independent_representative_projection_and_paired_statistics',
                      representative_projection=recomputed, representative_max_error=max(errors.values()),
                      native_evidence_tasks=1588, native_evidence_state_references=6773,
                      source_bound_last_position_first3_comparisons=778*3,
                      representative_full_first3_comparisons=2*3,
                      active_unique_prefixes=active_unique, unique_prefixes=421,
                      active_position_uses=active_uses, position_uses=548,
                      counts=counts, pairs=pairs, retention=retention,
                      strong_screen=strong, cost_review_eligible=cost,
                      label_scope='Uses only labels and historical predictions already released by the completed frozen analyzer; no new dataset label or heldout row was loaded.',
                      limitation='Reused exploratory 24 questions, not a full ARC result or generalization test. All 2334 first-three-loop last-position pairs are reread by the frozen numeric gate; full-sequence comparisons are limited to two representatives.')
    with (ROOT / 'results/r30_ROOT_POSTRUN_VERIFICATION.json').open('x', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
        f.write('\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
