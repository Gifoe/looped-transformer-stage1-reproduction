"""Independent arithmetic audit of previously completed E3 L-only endpoints.

No model execution, new selection, or causal claim is made by this audit.
"""
import collections
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rowfile = ROOT / 'PRIOR_CONDITIONAL_L_HOLDOUT_ROWS.jsonl'
sumfile = ROOT / 'PRIOR_CONDITIONAL_L_HOLDOUT_SUMMARY.json'
rows = [json.loads(x) for x in rowfile.read_text(encoding='utf-8').splitlines()]
summary = json.loads(sumfile.read_text(encoding='utf-8'))
seen = set()
max_gap_error = 0.
max_contrast_error = 0.
group_counts = collections.defaultdict(collections.Counter)
rescue_values = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    key = r['split'], r['id']
    assert key not in seen, key
    seen.add(key)
    gold = r['gold']
    k1, k2 = r['native_K1_prediction'], r['native_K2_prediction']
    group = ('stable_correct' if k1 == gold else 'rescue') if k2 == gold else ('damage' if k1 == gold else 'stable_wrong')
    assert group == r['group'], key
    group_counts[r['split']][group] += 1
    gaps = {}
    for name, c in r['conditions'].items():
        logits = [F(x) for x in c['candidate_logits']]
        ties = [i for i, v in enumerate(logits) if v == max(logits)]
        assert ties == c['ties'] and ties[0] == c['prediction'], (key, name)
        assert c['correct'] == (ties[0] == gold), (key, name)
        gap = logits[gold] - sum(v for i, v in enumerate(logits) if i != gold) / (len(logits)-1)
        assert gap == F(c['fraction_allwrong']), (key, name)
        max_gap_error = max(max_gap_error, abs(float(gap)-c['allwrong_gap']))
        gaps[name] = gap
    contrasts = {}
    for ctrl, suffix in [('main', ''), ('norm', '_norm'), ('unrelated', '_unrelated')]:
        contrasts['rest'+suffix] = gaps['P_plus_L_'+ctrl] - gaps['P']
        contrasts['loss'+suffix] = gaps['N'] - gaps['N_minus_L_'+ctrl]
    for op in ['rest', 'loss']:
        contrasts[op+'_specific'] = contrasts[op] - contrasts[op+'_unrelated']
    for name, value in contrasts.items():
        assert value == F(r['fraction_contrasts'][name]), (key, name)
        err = abs(float(value)-r['contrasts'][name])
        max_contrast_error = max(max_contrast_error, err)
        assert err < 2e-6, (key, name, err)
        if group == 'rescue':
            rescue_values[r['split']][name].append(value)

out = {'status': 'PASS', 'scope': 'Arithmetic and coverage audit of prior derived endpoint rows; not raw tensor/source or bootstrap replication.',
       'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [rowfile, sumfile]},
       'unique_question_rows': len(rows), 'group_counts': dict(group_counts),
       'max_reported_gap_error': max_gap_error, 'max_score_vs_exact_logit_contrast_error': max_contrast_error,
       'rescue_exact_means': {}}
for split, counts in group_counts.items():
    assert counts == summary[split]['group_counts']
    assert sum(counts.values()) == summary[split]['n']
    out['rescue_exact_means'][split] = {}
    for name, values in rescue_values[split].items():
        mean = sum(values)/len(values)
        if name in summary[split]['claims']:
            claim = summary[split]['claims'][name]
            assert mean == F(claim['fraction_exact_mean']), (split, name)
            assert abs(float(mean)-claim['mean']) < 2e-6, (split, name)
        out['rescue_exact_means'][split][name] = str(mean)
(ROOT/'PRIOR_L_INDEPENDENT_ARITHMETIC_AUDIT.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
print(json.dumps(out, indent=2))
