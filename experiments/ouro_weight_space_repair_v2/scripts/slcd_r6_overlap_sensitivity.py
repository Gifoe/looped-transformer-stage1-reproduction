"""Posthoc sensitivity to official split question-text overlap.

Official 1172-row result is unchanged. No method selection or new model call.
This does not remove semantic overlap or pretraining contamination.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]


def rows(path):
    with (ROOT/path).open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fingerprint(row):
    return ' '.join(row['question'].casefold().split())


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def main():
    acceptance=json.loads((ROOT/'results/slcd_R6_TEST_ACCEPTANCE.json').read_text())
    assert acceptance['status']=='passed' and acceptance['correct']==736
    cohorts={s:rows('data/slcd_'+s+'.jsonl') for s in ['discovery','selection','confirmation','validation','test']}
    fingerprints={s:{fingerprint(row) for row in data} for s,data in cohorts.items()}
    training=set.union(*(fingerprints[s] for s in ['discovery','selection','confirmation']))
    with (ROOT/'results/slcd_r6_paired_transitions.csv').open(newline='',encoding='utf8') as stream:
        raw=list(csv.DictReader(stream))
    methods=['native4','fixed','adaptive','incumbent_A1','r6_prior_projection']
    pred={name:{r['id']:int(r['prediction']) for r in raw if r['method']==name} for name in methods}
    annotated=[]
    for row in cohorts['test']:
        q=fingerprint(row)
        annotated.append(dict(id=row['id'],gold=row['gold'],training_overlap=q in training,
                              validation_overlap=q in fingerprints['validation'],
                              **{name:pred[name][row['id']] for name in methods}))
    results=[]
    for name,keep in [('official_all_1172',lambda r:True),
                      ('posthoc_exclude_training_question_overlap',lambda r:not r['training_overlap']),
                      ('posthoc_exclude_train_and_validation_question_overlap',lambda r:not (r['training_overlap'] or r['validation_overlap']))]:
        selected=[r for r in annotated if keep(r)];n=len(selected)
        for method in methods:
            for baseline in ['fixed','incumbent_A1']:
                good=[r[method]==r['gold'] for r in selected];base=[r[baseline]==r['gold'] for r in selected]
                rescues=sum(a and not b for a,b in zip(good,base));damage=sum(b and not a for a,b in zip(good,base));nd=rescues+damage
                p=1. if nd==0 else min(1.,2*sum(math.comb(nd,k) for k in range(min(rescues,damage)+1))/(2**nd))
                results.append(dict(scope=name,method=method,baseline=baseline,n=n,correct=sum(good),accuracy=sum(good)/n,
                                    corrections=rescues,regressions=damage,net_correct=rescues-damage,delta_pp=100*(rescues-damage)/n,exact_mcnemar_p=p))
    def csvout(path,data):
        with (ROOT/path).open('w',newline='',encoding='utf8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    csvout('results/slcd_r6_overlap_sensitivity.csv',results)
    csvout('results/slcd_r6_overlap_question_flags.csv',annotated)
    output=dict(status='complete_posthoc',official_result_unchanged=True,method_selection=False,
                training_overlap=sum(r['training_overlap'] for r in annotated),validation_overlap=sum(r['validation_overlap'] for r in annotated),
                union_overlap=sum(r['training_overlap'] or r['validation_overlap'] for r in annotated),
                fingerprint='casefold then collapse whitespace; question only',
                limitation='Exact-text sensitivity only, no semantic overlap or pretraining contamination audit; excluded subsets are posthoc and not a new blind benchmark.',
                primary_sensitivity=[r for r in results if r['method']=='r6_prior_projection'],
                source_sha256=sha('scripts/slcd_r6_overlap_sensitivity.py'),
                input_sha256={p:sha(p) for p in ['results/slcd_R6_TEST_ACCEPTANCE.json','results/slcd_r6_paired_transitions.csv']+['data/slcd_'+s+'.jsonl' for s in cohorts]})
    (ROOT/'results/slcd_R6_OVERLAP_SENSITIVITY.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
