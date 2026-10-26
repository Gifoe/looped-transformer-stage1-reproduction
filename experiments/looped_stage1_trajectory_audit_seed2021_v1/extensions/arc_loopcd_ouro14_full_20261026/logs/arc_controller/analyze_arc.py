"""Independent complete-set/score audit and precommitted paired replication gate."""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np
from scipy.stats import binomtest
from common import ROOT,BASE,CFG,PROTOCOL_SHA,read,save


def main():
    docs=read('data/arc_full_test.jsonl');rows=read('trajectories/arc_three_methods.jsonl')
    assert len(docs)==len(rows)==1172
    assert [r['id'] for r in rows]==[r['id'] for r in docs]
    files=[ROOT/'scripts/common.py',ROOT/'scripts/methods.py',BASE/'scripts/qwen_recipe.py',ROOT/'scripts/score_arc.py']
    fingerprint=hashlib.sha256(''.join(p.name+hashlib.sha256(p.read_bytes()).hexdigest() for p in files).encode()).hexdigest()
    parity=json.loads((ROOT/'results/ARC_PARITY.json').read_text())
    assert parity['passed'] and parity['protocol_sha256']==PROTOCOL_SHA
    assert json.loads((ROOT/'results/ARC_WORKER_COMPLETE.json').read_text())['passed']
    correct={m:[] for m in CFG['arc_methods']};raw_correct={m:[] for m in CFG['arc_methods']};transitions=[]
    for d,r in zip(docs,rows):
        assert r['protocol_sha256']==PROTOCOL_SHA and r['implementation_sha256']==fingerprint
        assert r['model_revision']==CFG['model_revision'] and r['model']=='Ouro-1.4B'
        assert r['fewshot_count']==25 and r['recurrent_depth']==4 and r['reference_depth']==1
        assert r['prompt_sha256']==d['prompt_sha256'] and r['gold']==d['gold']
        assert set(r['methods'])==set(CFG['arc_methods'])
        assert r['fixed_omega']==.5 and r['adaptive_cap']==1.
        assert 0<=r['adaptive_strength_min']<=r['adaptive_strength_max']<=1.
        assert math.isfinite(r['elapsed_s']) and r['elapsed_s']>0
        for method,result in r['methods'].items():
            values=np.asarray(result['sum_scores']);lengths=np.asarray(d['choice_char_lengths'])
            assert values.shape==lengths.shape and np.isfinite(values).all() and (lengths>0).all()
            normalized=values/lengths
            assert np.allclose(normalized,result['normalized_scores'],atol=1e-12,rtol=1e-12)
            prediction=int(normalized.argmax());raw=int(values.argmax())
            assert prediction==result['prediction']==result['normalized_prediction']
            assert raw==result['raw_prediction']
            assert result['correct']==result['normalized_correct']==(prediction==d['gold'])
            correct[method].append(prediction==d['gold']);raw_correct[method].append(raw==d['gold'])
    baseline=np.asarray(correct['baseline'],dtype=int)
    rng=np.random.default_rng(CFG['bootstrap_seed'])
    draws=rng.integers(0,1172,size=(CFG['bootstrap_resamples'],1172))
    summary=[];paired={}
    for method in CFG['arc_methods']:
        a=np.asarray(correct[method],dtype=int);delta=a-baseline
        fixed=int(((baseline==0)&(a==1)).sum());harmed=int(((baseline==1)&(a==0)).sum())
        assert fixed-harmed==int(delta.sum())
        bootstrap=delta[draws].mean(axis=1)*100
        low,high=np.quantile(bootstrap,[.025,.975])
        test=1. if fixed+harmed==0 else float(binomtest(fixed,fixed+harmed,.5).pvalue)
        item=dict(method=method,n=1172,correct=int(a.sum()),accuracy_percent=float(a.mean()*100),
            paper_accuracy_percent=CFG['paper_arc_accuracy_percent'][method],
            accuracy_error_pp=float(a.mean()*100-CFG['paper_arc_accuracy_percent'][method]),
            corrections=fixed,degradations=harmed,net_corrections=fixed-harmed,delta_pp=float(delta.mean()*100),
            paired_bootstrap_95ci_low_pp=float(low),paired_bootstrap_95ci_high_pp=float(high),
            paired_ci_excludes_zero=bool(low>0 or high<0),
            mcnemar_exact_two_sided_p=test,secondary_unnormalized_correct=int(sum(raw_correct[method])),
            secondary_unnormalized_accuracy_percent=float(np.mean(raw_correct[method])*100))
        summary.append(item)
        if method!='baseline':
            paired[method]=item
            for d,r,b,v in zip(docs,rows,baseline,a):
                transitions.append(dict(id=d['id'],method=method,baseline_prediction=r['methods']['baseline']['prediction'],
                    guided_prediction=r['methods'][method]['prediction'],gold=d['gold'],baseline_correct=int(b),guided_correct=int(v),
                    transition='correction' if not b and v else 'degradation' if b and not v else 'both_correct' if b else 'both_wrong'))
    for name,records in [('arc_summary.csv',summary),('arc_paired_transitions.csv',transitions)]:
        with (ROOT/'results'/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    criterion=CFG['arc_acceptance']
    absolute=all(abs(s['accuracy_error_pp'])<=criterion['absolute_accuracy_tolerance_pp'] for s in summary)
    effects={m:dict(net_positive=s['net_corrections']>0,
        delta_within_tolerance=abs(s['delta_pp']-CFG['paper_arc_delta_percent'][m])<=criterion['paired_delta_tolerance_pp'],
        reference_delta_in_ci=s['paired_bootstrap_95ci_low_pp']<=CFG['paper_arc_delta_percent'][m]<=s['paired_bootstrap_95ci_high_pp']) for m,s in paired.items()}
    passed=absolute and all(all(v.values()) for v in effects.values())
    result=dict(technical_audit_passed=True,replication_gate_passed=passed,protocol_sha256=PROTOCOL_SHA,
        implementation_sha256=fingerprint,all1172_same_inputs=True,absolute_accuracy_condition=absolute,
        paired_effect_conditions=effects,criteria_frozen_in_protocol=criterion,summary=summary,
        raw_sha256=hashlib.sha256((ROOT/'trajectories/arc_three_methods.jsonl').read_bytes()).hexdigest(),
        inference_note='Positive net corrections and agreement with reported effect size are separate from evidence of statistical significance. Original gross correction/degradation counts cannot be reconstructed from paper aggregate accuracy.',
        gsm8k_allowed=passed,decision='Proceed to full1319-question GSM8K under frozen paper settings' if passed else 'Do not start GSM8K; report this ARC reproduction discrepancy without test tuning')
    save('results/ARC_REPLICATION_GATE.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
