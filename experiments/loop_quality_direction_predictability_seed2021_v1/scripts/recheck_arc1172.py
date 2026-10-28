"""Independent no-GPU reconstruction from candidate score vectors and IDs."""
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from quality_core import BASE,ROOT,read,put,digest


def main():
    a=BASE/'extensions/arc_loopcd_ouro14_full_20261026'
    raw=read(a/'trajectories/arc_three_methods.jsonl');docs=read(a/'data/arc_full_test.jsonl');byid={x['id']:x for x in docs}
    assert len(raw)==len(byid)==1172 and len({r['id'] for r in raw})==1172
    correct={k:[] for k in ['baseline','fixed','adaptive']}
    for r in raw:
        d=byid[r['id']];assert d['gold']==r['gold'] and d['prompt_sha256']==r['prompt_sha256']
        for k in correct:
            prediction=int(np.argmax(np.array(r['methods'][k]['sum_scores'])/np.array(d['choice_char_lengths'])))
            assert prediction==r['methods'][k]['prediction'];correct[k].append(int(prediction==d['gold']))
    reference=pd.read_csv(a/'results/arc_summary.csv').set_index('method');rows=[]
    boot=np.random.default_rng(2021).integers(0,1172,(10000,1172))
    for k in correct:
        v=np.array(correct[k]);b=np.array(correct['baseline']);d=v-b
        w=int((d>0).sum());loss=int((d<0).sum());ci=np.quantile(d[boot].mean(1)*100,[.025,.975])
        p=binomtest(w,w+loss,.5).pvalue if w+loss else 1.;ref=reference.loc[k]
        assert int(v.sum())==ref.correct and w==ref.corrections and loss==ref.degradations
        assert abs(p-ref.mcnemar_exact_two_sided_p)<1e-12
        assert np.allclose(ci,[ref.paired_bootstrap_95ci_low_pp,ref.paired_bootstrap_95ci_high_pp],atol=1e-12)
        rows.append(dict(method=k,correct=int(v.sum()),n=1172,accuracy_percent=v.mean()*100,corrections=w,degradations=loss,
                         delta_pp=d.mean()*100,ci95_pp=ci.tolist(),mcnemar_p=p))
    put('results/ARC1172_DELIVERY_RECHECK.json',dict(passed=True,questions=1172,methods=3,
        raw_sha256=digest(a/'trajectories/arc_three_methods.jsonl'),source='Separate CPU recomputation directly from raw candidate score vectors, ID joins and answer lengths; no producer/auditor import',
        summary=rows,new_gpu_hours=0))
    print('ARC1172 independent recheck PASS',[(x['method'],x['correct']) for x in rows])


if __name__=='__main__':main()
