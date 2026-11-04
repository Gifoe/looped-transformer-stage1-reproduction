"""Post-freeze ARC score audit and descriptive cases; never selects a method.

Run after R6_TEST_ACCEPTANCE. Only the frozen selected rule is evaluated.
No alternate test coefficients, label-based thresholds, or new model calls.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from slcd_r6_scores import selected_score, projection_parameters


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf8'))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def records(path):
    with (ROOT / path).open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path, rows):
    with (ROOT / path).open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def margin(scores, gold):
    return float(scores[gold] - max(np.delete(scores, gold)))


def main():
    lock = read('configs/slcd_R6_FINAL_LOCK.json')
    receipt = read('results/slcd_R6_TEST_ACCEPTANCE.json')
    assert receipt['status'] == 'passed'
    assert digest('configs/slcd_R6_FINAL_LOCK.json') == receipt['final_lock_sha256']
    assert digest('scripts/slcd_r6_scores.py') == lock['ranking_source_sha256']
    assert digest('results/slcd_original_test.jsonl') == lock['original_test_scores_sha256']
    prior_receipt = read('results/slcd_r6_priors_test_COMPLETE.json')
    assert digest('results/slcd_r6_priors_test.jsonl') == prior_receipt['score_file_sha256']
    assert digest('data/slcd_test.jsonl') == lock['data_sha256']['test']
    originals = records('results/slcd_original_test.jsonl')
    priors = {r['id']: r for r in records('results/slcd_r6_priors_test.jsonl')}
    manifest = {r['id']: r for r in records('data/slcd_test.jsonl')}
    assert len(originals) == len(priors) == len(manifest) == 1172
    candidates, questions, cases = [], [], []
    for row in originals:
        qid = row['id']; prior = priors[qid]; item = manifest[qid]
        assert row['gold'] == prior['gold'] == item['gold']
        assert row['prompt_sha256'] == prior['prompt_sha256'] == item['prompt_sha256']
        assert row['choice_char_lengths'] == prior['choice_char_lengths'] == item['choice_char_lengths']
        m = row['methods']; pm = prior['prior_methods']; gold = row['gold']
        vec = lambda methods, key: np.asarray(methods[key]['normalized_scores'], dtype=np.float64)
        f = vec(m, 'loopcd_k4_fixed'); k4 = vec(m, 'native_k4'); k5 = vec(m, 'native_k5')
        a = f + (k4-k5); g = f-k4
        e = vec(pm, 'loopcd_k4_fixed') - vec(pm, 'native_k4')
        alpha, _, gc, ec = projection_parameters(g, e)
        score = selected_score(m, pm, lock['primary'])
        expected = a.copy() if alpha == 0 else a-alpha*e
        assert np.array_equal(score, expected)
        predictions = dict(native4=int(k4.argmax()), fixed=int(f.argmax()), incumbent_A1=int(a.argmax()), r6_prior_projection=int(score.argmax()))
        base = predictions['incumbent_A1']; new = predictions['r6_prior_projection']
        transition = ('unchanged' if base == new else 'rescue' if new == gold else 'damage' if base == gold else 'wrong_to_wrong')
        q = dict(id=qid, gold=gold, **predictions, transition_vs_A1=transition,
                 alpha=alpha, alpha_region='zero' if alpha == 0 else 'one' if alpha == 1 else 'interior',
                 guidance_norm=float(np.linalg.norm(gc)), prior_guidance_norm=float(np.linalg.norm(ec)),
                 A1_gold_margin=margin(a,gold), R6_gold_margin=margin(score,gold),
                 delta_gold_margin=margin(score,gold)-margin(a,gold), candidate_count=len(a))
        questions.append(q)
        per_choice = []
        for i, text in enumerate(item['choices']):
            c = dict(id=qid, choice_index=i, choice_text=text, gold=int(i==gold),
                     character_count=row['choice_char_lengths'][i], target_token_count=row['choice_token_counts'][i],
                     native4=float(k4[i]), native5=float(k5[i]), fixed=float(f[i]),
                     A1=float(a[i]), prior_native4=float(vec(pm,'native_k4')[i]),
                     prior_fixed=float(vec(pm,'loopcd_k4_fixed')[i]), guidance=float(g[i]),
                     prior_guidance=float(e[i]), alpha=alpha, correction=float(-alpha*e[i]), R6=float(score[i]))
            candidates.append(c); per_choice.append(c)
        if base != new:
            cases.append(dict(**q, question=item['question'], choices=per_choice,
                              interpretation='Descriptive score decomposition only; no causal hidden-state claim.'))
    assert sum(q['r6_prior_projection']==q['gold'] for q in questions)==receipt['correct']==736
    counts={name:sum(q['transition_vs_A1']==name for q in questions) for name in ['unchanged','rescue','damage','wrong_to_wrong']}
    assert (counts['rescue'],counts['damage'])==(23,13)
    write_csv('results/slcd_r6_arc_candidate_scores.csv', candidates)
    write_csv('results/slcd_r6_arc_question_diagnostics.csv', questions)
    with (ROOT/'results/slcd_r6_arc_changed_cases.jsonl').open('w',encoding='utf8') as stream:
        for case in cases:stream.write(json.dumps(case,ensure_ascii=False,allow_nan=False)+'\n')
    summary=dict(status='passed',n=1172,correct=736,candidate_rows=len(candidates),
                 transition_counts_vs_A1=counts,alpha_regions={name:sum(q['alpha_region']==name for q in questions) for name in ['zero','interior','one']},
                 regions=[dict(region=name,n=sum(q['alpha_region']==name for q in questions),
                               rescues=sum(q['alpha_region']==name and q['transition_vs_A1']=='rescue' for q in questions),
                               damage=sum(q['alpha_region']==name and q['transition_vs_A1']=='damage' for q in questions)) for name in ['zero','interior','one']],
                 frozen_primary=lock['primary'], source_sha256=digest('scripts/slcd_r6_case_audit.py'),
                 posthoc=True,method_selection=False,scope='Only selected frozen rule. No new model forwards, alternatives or causal hidden-state interventions.',
                 input_sha256={p:digest(p) for p in ['configs/slcd_R6_FINAL_LOCK.json','scripts/slcd_r6_scores.py','results/slcd_original_test.jsonl','results/slcd_r6_priors_test.jsonl','data/slcd_test.jsonl']})
    (ROOT/'results/slcd_R6_CASE_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    lines=['# Frozen R6 ARC changed cases', '',
           'Posthoc decomposition of the selected frozen rule. These cases were not used to change coefficients or choose another method.',
           '', f"All changes versus A1: {counts}. Complete vectors: [candidate CSV](slcd_r6_arc_candidate_scores.csv); all changed cases: [JSONL](slcd_r6_arc_changed_cases.jsonl).", '',
           'The score intervention establishes which ranking component flipped a decision. It does not identify a hidden-state causal mechanism, factual knowledge source, or calibrated probability.', '']
    # Show the smallest source-index rescue and damage, not the most favorable examples.
    for kind in ['rescue','damage']:
        case=next(c for c in cases if c['transition_vs_A1']==kind)
        lines.extend([f"## First {kind} in source order: {case['id']}", '', case['question'], '',
                      f"Gold {case['gold']}; A1 {case['incumbent_A1']}; R6 {case['r6_prior_projection']}; alpha={case['alpha']:.6g}. Indices are zero-based.", '',
                      '| Choice | Text | A1 | Prior guidance e | Correction -alpha*e | R6 |',
                      '|---|---|---:|---:|---:|---:|'])
        for c in case['choices']:
            text=c['choice_text'].replace('|','\\|').replace('\n',' ')
            lines.append(f"| {c['choice_index']} | {text} | {c['A1']:.6f} | {c['prior_guidance']:.6f} | {c['correction']:.6f} | {c['R6']:.6f} |")
        lines.extend(['',f"Gold margin {case['A1_gold_margin']:.6f} -> {case['R6_gold_margin']:.6f}.", ''])
    (ROOT/'results/slcd_R6_FAILURE_CASES.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
