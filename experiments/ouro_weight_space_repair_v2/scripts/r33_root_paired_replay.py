"""Independent already-unsealed paired arithmetic and lifecycle audit; CPU only."""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def main():
    spec = read('configs/r33_COMMON_GAIN_PILOT_LOCK.json')
    for group in ('source_sha256','input_sha256'):
        for path, expected in spec[group].items():
            p = ROOT/path
            if p.exists():
                assert sha(path) == expected, path
            else:
                assert path.startswith('models/Ouro-1.4B/'), path
    ledgers = spec['budget']['inherited_ledger_sha256']
    assert len(ledgers) == 18
    for path, expected in ledgers.items():
        assert sha(path) == expected
    prior = sum(read(path)['used_seconds'] for path in ledgers)
    assert prior == 20755.46525352029
    book, sup = read('logs/r33_gpu_budget.json'), read('logs/r33_SUPERVISOR.json')
    assert sup['status'] == 'complete' and sup['exit_code'] == 0
    assert not sup.get('timed_out', False) and not sup['overrun'] and 0 < sup['gpu_worker_seconds'] <= 200
    assert book['used_seconds'] == sup['gpu_worker_seconds'] and book['inherited_total_seconds'] == prior
    assert book['total_used_seconds'] == prior+book['used_seconds'] <= 21600
    c = read('results/r33_COMPLETE.json')
    assert c['status'] == 'complete' and c['n'] == 24 and c['groups'] == 389
    assert c['body_calls'] == 1 and c['body_heads'] == 3 and c['native_head_calls'] == 7194 and c['total_heads'] == 7197
    assert len(c['all_parameter_hashes_before']) == 269
    assert c['all_parameter_hashes_before'] == c['all_parameter_hashes_after'] == read('results/r28_COMPLETE.json')['all_parameter_hashes_after']
    for key,path in [('scores_sha256','results/r33_SCORES.json'),
                     ('audit_sha256','results/r33_EXECUTION_AUDIT.json'),
                     ('representative_vectors_sha256','results/r33_REPRESENTATIVE_VECTORS.npz'),
                     ('native_parity_sha256','results/r33_NATIVE_PARITY.json')]:
        assert c[key] == sha(path)
    analysis = read('results/r33_ANALYSIS.json')
    assert analysis['protocol_sha256'] == c['protocol_sha256'] == sha('configs/r33_COMMON_GAIN_PILOT_LOCK.json')
    output = dict(status='passed_independent_lifecycle_ledger_and_output_identity',
                  new_GPU_seconds=0,model_loaded=False,actual_worker_seconds=book['used_seconds'],
                  inherited_ledger_count=18,total_GPU_seconds=book['total_used_seconds'],
                  remaining_seconds=21600-book['total_used_seconds'],unchanged_parameters=269,
                  protocol_sha256=c['protocol_sha256'])
    if analysis['status'] == 'complete_but_unmatched_no_effect_unsealing':
        output.update(status='passed_numeric_lifecycle_unmatched_no_effect_replay',unmatched=analysis['unmatched'],labels_opened=False)
    else:
        assert analysis['status'] == 'complete_analyzed_once'
        rows = read('results/r33_SCORES.json')
        predictions = {r['id']:r for r in analysis['predictions']}
        assert len(predictions) == len(rows['rows']) == 24
        methods = list(rows['rows'][0]['candidates'][0]['scores'])+['R6_historical','Fixed_historical']
        correct = {name:[] for name in methods}
        identities = []
        for row in rows['rows']:
            identity = row['id']; identities.append(identity)
            released = predictions[identity]; gold = released['gold']
            assert type(gold) is int and 0 <= gold < len(row['candidates'])
            for name in methods:
                if name.endswith('_historical'):
                    pred = released['predictions'][name]
                else:
                    values = [candidate['scores'][name] for candidate in row['candidates']]
                    assert all(math.isfinite(v) for v in values)
                    pred = max(range(len(values)),key=lambda i:values[i])
                    assert pred == released['predictions'][name], (identity,name)
                correct[name].append(pred == gold)
        counts = {name:sum(v) for name,v in correct.items()}
        assert counts == analysis['counts']
        loo = ['LOO_N0','LOO_N1','LOO_N2']
        pairs, retention = {}, {}
        for base,fixed in [('R6_historical','Fixed_historical'),('R6_matched','Fixed')]:
            pairs[base] = {}
            for name in methods:
                rescue = sum(a and not b for a,b in zip(correct[name],correct[base]))
                damage = sum(b and not a for a,b in zip(correct[name],correct[base]))
                pairs[base][name] = dict(rescue=rescue,damage=damage,net=rescue-damage)
            special = [i for i,(a,b) in enumerate(zip(correct[base],correct[fixed])) if a and not b]
            retention[base] = {name:dict(total=len(special),retained=sum(correct[name][i] for i in special),
                                         ids=[identities[i] for i in special]) for name in ['primary',*loo]}
        assert pairs == analysis['pairs'] and retention == analysis['retention']
        controls = ['M3','J2_guidance','G_matched','vocabulary_null','matched_scale','reference_consensus_matched']
        assert controls == spec['strong_controls']
        protected = all(retention[base]['primary']['retained'] == retention[base]['primary']['total'] for base in retention)
        cost = protected and all(pairs[base]['primary']['net'] >= 0 for base in pairs)
        strong = protected and all(pairs[base]['primary']['rescue'] >= 2 and pairs[base]['primary']['damage'] == 0 for base in pairs)
        strong = strong and all(counts['primary'] > counts[name] for name in controls)
        strong = strong and all(pairs[base][name]['net'] >= 0 and retention[base][name]['retained'] == retention[base][name]['total']
                                for base in pairs for name in loo)
        assert strong == analysis['strong_screen'] and cost == analysis['cost_review_eligible']
        diagnostics = rows['diagnostics']
        graph = read('configs/r28_TRUE_PREFIX_TASK_INVENTORY.json')
        active = sum(not d['h_exact_zero'] for d in diagnostics.values())
        uses = sum(not diagnostics[u['tasks']['Q']]['h_exact_zero'] for u in graph['usages'])
        output.update(status='passed_independent_all24_scores_pairs_retention_and_gates',counts=counts,pairs=pairs,
                      retention=retention,strong_screen=strong,cost_review_eligible=cost,
                      active_unique_prefixes=active,total_unique_prefixes=len(diagnostics),
                      active_position_uses=uses,total_position_uses=len(graph['usages']),
                      primary_prediction_agreement={name:sum(r['predictions']['primary'] == r['predictions'][name]
                                                            for r in predictions.values()) for name in methods},
                      label_scope='Only the same 24 labels and historical predictions already released by the clean frozen analyzer; no new labels or heldout data opened.')
    output['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output['limitation'] = 'Lifecycle and already-unsealed scalar arithmetic, not independent all-prefix fullV recomputation or a full ARC/generalization result.'
    with (ROOT/'results/r33_ROOT_PAIRED_REPLAY.json').open('x',encoding='utf-8') as f:
        json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
