"""Describe every already-unsealed R6-relative damage; no new inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
from r28_prefix_task_inventory import selected

ROOT = Path(__file__).resolve().parents[1]


def main():
    analysis = json.loads((ROOT/'results/r28_ANALYSIS.json').read_text())
    assert analysis['status'] == 'complete_analyzed_once'
    scores = {r['id']: r for r in json.loads((ROOT/'results/r28_SCORES.json').read_text())['rows']}
    requests = {r['id']: r for r in json.loads((ROOT/'configs/r28_SANITIZED_REQUESTS.json').read_text())['rows']}
    damaged = {r['id']: r for r in analysis['predictions']
               if r['predictions']['R6_matched'] == r['gold'] and r['predictions']['primary'] != r['gold']}
    questions = {}
    for split in ('discovery', 'selection'):
        with (ROOT/('data/slcd_'+split+'.jsonl')).open(encoding='utf-8') as f:
            for line in f:
                identity = selected(line, {'id'})['id']
                if identity in damaged:
                    questions[identity] = selected(line, {'source_doc'})['source_doc']['question']
    cases = []
    for qid, record in damaged.items():
        row = scores[qid]
        gold, wrong = record['gold'], record['predictions']['primary']
        vectors = {m: np.asarray([c['scores'][m] for c in row['candidates']], dtype=np.float64)
                   for m in ('primary', 'A1', 'R6_matched', 'M2', 'M3', 'J2_guidance', 'matched_scale')}
        margins = {m: float(v[gold]-v[wrong]) for m, v in vectors.items()}
        candidates = []
        for j, c in enumerate(row['candidates']):
            primary = np.asarray(c['pieces']['primary']['target_log_probabilities'])
            fixed = np.asarray(c['pieces']['Fixed']['target_log_probabilities'])
            candidates.append(dict(index=j, text=requests[qid]['candidates'][j]['request_pair'][1],
                                   original_characters=c['characters'], scores={m: float(v[j]) for m, v in vectors.items()},
                                   primary_minus_A1_score=float(vectors['primary'][j]-vectors['A1'][j]),
                                   per_token_primary_minus_Fixed_logprob=(primary-fixed).tolist()))
        cases.append(dict(id=qid, question=questions[qid], gold_index=gold, primary_wrong_index=wrong,
                          primary_same_as_A1_prediction=record['predictions']['primary'] == record['predictions']['A1'],
                          gold_minus_primary_wrong_margins=margins,
                          primary_vs_A1_margin_change=margins['primary']-margins['A1'],
                          R6_vs_A1_margin_change=margins['R6_matched']-margins['A1'],
                          candidates=candidates))
    assert len(cases) == analysis['pairs']['R6_matched']['primary']['damage'] == 1
    result = dict(status='all_unsealed_R6_damage_cases_described', cases=cases,
                  inference_or_new_GPU=False,
                  scope='Reports observed scalar log-probability changes. Does not separate target-logit and log-normalizer effects, identify semantic causation, or tune a new formula.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (ROOT/'results/r28_ROOT_FAILURE_CASES.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
