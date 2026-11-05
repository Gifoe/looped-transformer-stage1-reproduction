"""All already-unsealed pilot rows, without extra inference or rule selection."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def main():
    current = read('results/r30_ANALYSIS.json')
    prior = read('results/r28_ANALYSIS.json')
    assert current['status'] == prior['status'] == 'complete_analyzed_once'
    current_rows = read('results/r30_SCORES.json')['rows']
    old_rows = {r['id']: r for r in read('results/r28_SCORES.json')['rows']}
    labels = {r['id']: r for r in current['predictions']}
    old_labels = {r['id']: r for r in prior['predictions']}
    requests = {r['id']: r for r in read('configs/r28_SANITIZED_REQUESTS.json')['rows']}
    assert len(current_rows) == len(labels) == 24
    assert {r['id'] for r in current_rows} == set(old_rows) == set(labels) == set(old_labels)
    cases, c_changes, a_changes = [], [], []
    for row in current_rows:
        qid = row['id']
        gold = labels[qid]['gold']
        assert gold == old_labels[qid]['gold'] and len(row['candidates']) == 4
        names = tuple(row['candidates'][0]['scores'])
        vectors = {name: np.asarray([c['scores'][name] for c in row['candidates']], dtype=np.float64)
                   for name in names}
        vectors['R28_primary'] = np.asarray([c['scores']['primary'] for c in old_rows[qid]['candidates']], dtype=np.float64)
        assert all(np.isfinite(v).all() for v in vectors.values())
        for name in ('A1', 'R6_matched', 'Fixed', 'M2', 'M3', 'J2_guidance'):
            assert np.array_equal(vectors[name], np.asarray([c['scores'][name] for c in old_rows[qid]['candidates']]))
        predictions = {name: int(np.argmax(v)) for name, v in vectors.items()}
        assert all(predictions[name] == labels[qid]['predictions'][name] for name in names)
        assert predictions['R28_primary'] == old_labels[qid]['predictions']['primary']
        wrong = [j for j in range(4) if j != gold]
        margins = {name: float(v[gold] - max(v[j] for j in wrong)) for name, v in vectors.items()}
        c_delta = vectors['primary'] - vectors['A1']
        a_delta = vectors['R28_primary'] - vectors['A1']
        c_centered, a_centered = c_delta-c_delta.mean(), a_delta-a_delta.mean()
        c_changes.extend(c_centered.tolist()); a_changes.extend(a_centered.tolist())
        candidates = [dict(index=j, text=requests[qid]['candidates'][j]['request_pair'][1],
                           original_characters=c['characters'],
                           scores={name: float(v[j]) for name, v in vectors.items()},
                           clock_primary_minus_A1=float(c_delta[j]),
                           spatial_primary_minus_A1=float(a_delta[j]),
                           clock_minus_spatial=float(c_delta[j]-a_delta[j]))
                      for j, c in enumerate(row['candidates'])]
        cases.append(dict(id=qid, gold_index=gold, predictions=predictions,
                          correctness={name: pred == gold for name, pred in predictions.items()},
                          gold_minus_best_wrong_margin=margins,
                          clock_vs_A1_margin_change=margins['primary']-margins['A1'],
                          clock_vs_spatial_margin_change=margins['primary']-margins['R28_primary'],
                          clock_vs_R6_margin_change=margins['primary']-margins['R6_matched'],
                          candidates=candidates))

    def summarize(key):
        values = [case[key] for case in cases]
        return dict(increased=sum(v > 0 for v in values), unchanged=sum(v == 0 for v in values),
                    decreased=sum(v < 0 for v in values), mean=math.fsum(values)/len(values),
                    median=float(np.median(values)), minimum=min(values), maximum=max(values))

    c, a = np.asarray(c_changes), np.asarray(a_changes)
    norm = float(np.linalg.norm(c)*np.linalg.norm(a))
    agreement = {name: sum(case['predictions']['primary'] == case['predictions'][name] for case in cases)
                 for name in cases[0]['predictions']}
    result = dict(status='all24_complete_margin_inventory', rows=24,
                  candidate_rows=sum(len(case['candidates']) for case in cases),
                  inference_or_new_GPU=False, model_loaded=False,
                  label_scope='Only the same 24 labels already released by the two frozen analyzers; no other dataset or heldout record opened.',
                  interpretation='Descriptive scalar score inventory. Not a novel estimator, efficacy test on independent data, causal attribution, or promotion of a control.',
                  gold_margin_changes={key:summarize(key) for key in ('clock_vs_A1_margin_change', 'clock_vs_spatial_margin_change', 'clock_vs_R6_margin_change')},
                  primary_prediction_agreement=agreement,
                  centered_candidate_correction_cosine_clock_vs_spatial=None if norm == 0 else float(np.dot(c,a)/norm),
                  centered_candidate_correction_norm_clock=float(np.linalg.norm(c)),
                  centered_candidate_correction_norm_spatial=float(np.linalg.norm(a)),
                  counts=current['counts'], strong_screen=current['strong_screen'],
                  cost_review_eligible=current['cost_review_eligible'], cases=cases,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  inputs_sha256={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in (
                      'results/r30_ANALYSIS.json', 'results/r30_SCORES.json',
                      'results/r28_ANALYSIS.json', 'results/r28_SCORES.json',
                      'configs/r28_SANITIZED_REQUESTS.json')})
    with (ROOT/'results/r30_ROOT_ALL24_MARGIN_INVENTORY.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
