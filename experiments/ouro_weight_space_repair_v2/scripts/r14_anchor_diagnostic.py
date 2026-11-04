"""One fixed single-reference anchored refinement, cached development only.

Same N0, prefix positions, vocabulary measure and normalized LL implementation
for M2 and the fixed whole-alpha bridge. No learned coefficient or new evidence.
No multi-reference proxy, strength search, model call or heldout/test reader.
"""
import argparse,csv,json,math
import numpy as np
from r7_core import ROOT,read,sha,jsonlines
from r8_scores import center,norm,finite
from slcd_r6_scores import selected_score,_inputs

LOCK='configs/r14_ANCHOR_DIAGNOSTIC_PROTOCOL.json'
MAIN='anchored_token_refinement'


def compute(a,r,e,m,bridge):
    a,r,e,m,bridge=[np.asarray(v,dtype=np.float64) for v in [a,r,e,m,bridge]]
    assert all(v.ndim==1 and v.shape==a.shape for v in [a,r,e,m,bridge]);finite(a,r,e,m,bridge)
    delta=m-bridge;new=r+delta;size=norm(center(new-a));en=norm(center(e));possible=en>0 or size==0.
    controls={MAIN:new,'reversed_refinement':r-delta,'prior_norm':a.copy() if en==0 else a-(size/en)*e,
        'shuffled_refinement':r+center(delta)[np.random.default_rng(2021).permutation(len(a))]}
    assert np.array_equal(r+(bridge-bridge),r)
    finite(*controls.values())
    return controls,dict(norm_match_possible=possible,final_correction_norm=size,prior_norm_error=abs(norm(center(controls['prior_norm']-a))-size),
        refinement_norm=norm(center(delta)),shuffled_refinement_norm=norm(center(controls['shuffled_refinement']-r)))


def paired(data,name,reference):
    x=[r['correct'][name] for r in data];b=[r['correct'][reference] for r in data]
    rescue=sum(a and not c for a,c in zip(x,b));damage=sum(not a and c for a,c in zip(x,b));discord=rescue+damage
    p=1. if not discord else min(1.,2*sum(math.comb(discord,k) for k in range(min(rescue,damage)+1))/2**discord)
    return dict(method=name,reference=reference,n=len(data),correct=sum(x),reference_correct=sum(b),net=rescue-damage,rescue=rescue,damage=damage,exact_mcnemar_p=p)


def main(phase):
    spec=read(LOCK);assert sha('scripts/r14_anchor_diagnostic.py')==spec['source_sha256'] and not spec['heldout_access_authorized'] and spec['GPU_seconds']==0
    for p,h in spec['inference_source_sha256'].items():assert sha('scripts/'+p)==h
    splits=['discovery','selection'] if phase=='development' else ['confirmation']
    if phase=='confirmation':assert read('results/r14_development_COMPLETE.json')['passes_gate']
    results=[]
    for split in splits:
        path=f'results/r7_{split}_development.jsonl';assert sha(path)==spec['input_sha256'][path]
        receipt=read(path.replace('.jsonl','_COMPLETE.json'));assert receipt['status']=='complete' and receipt['score_file_sha256']==sha(path)
        data=list(jsonlines(path));assert len(data)==len({r['id'] for r in data})==spec['split_sizes'][split]
        for row in data:
            assert row['signature']==receipt['signature'] and not any(row['alignment_fallback_by_method'][n] for n in ['M2_N0_token','C_R6alpha_token'])
            for d in row['token_diagnostics']:
                left,right=d['M2_N0_token'],d['C_R6alpha_token']
                for k in ['numerator','denominator','robust_denominator','guidance_norm2']:
                    assert np.array_equal(left[k],right[k]),('M2/bridge differ beyond alpha',row['id'],k)
                assert all(v==row['alpha6'] for v in right['alpha'])
            a,g,e=_inputs(row['original_methods'],row['prior_methods'][0]);r=np.asarray(row['baseline_scores']['M0_R6'])
            assert np.array_equal(r,selected_score(row['original_methods'],row['prior_methods'][0],'r6_prior_projection'))
            values,diag=compute(a,r,e,row['method_scores']['M2_N0_token'],row['method_scores']['C_R6alpha_token'])
            assert not diag['norm_match_possible'] or diag['prior_norm_error']<1e-12*max(diag['final_correction_norm'],1.)
            assert abs(diag['refinement_norm']-diag['shuffled_refinement_norm'])<1e-12*max(diag['refinement_norm'],1.)
            values.update(R6=r,A1=a,M2=np.asarray(row['method_scores']['M2_N0_token']),bridge=np.asarray(row['method_scores']['C_R6alpha_token']),fixed=np.asarray(row['baseline_scores']['fixed']))
            pred={k:int(np.argmax(v)) for k,v in values.items()}
            results.append(dict(id=row['id'],split=split,gold=row['gold'],scores={k:v.tolist() for k,v in values.items()},predictions=pred,correct={k:v==row['gold'] for k,v in pred.items()},diagnostics=diag,protocol_sha256=sha(LOCK)))
    path=f'results/r14_{phase}_scores.jsonl';assert not (ROOT/path).exists()
    with (ROOT/path).open('x',encoding='utf8') as f:
        for row in results:f.write(json.dumps(row,allow_nan=False)+'\n')
    if phase=='confirmation':
        dev=read('results/r14_development_COMPLETE.json');assert sha(dev['score_file'])==dev['score_file_sha256']
        pool=list(jsonlines(dev['score_file']))+results
    else:pool=results
    refs=['R6','M2','bridge','reversed_refinement','prior_norm','shuffled_refinement'];comparisons=[paired(pool,MAIN,k) for k in refs];by={r['reference']:r for r in comparisons}
    n=len(pool);expected=360 if phase=='development' else 705;assert sum(r['correct']['R6'] for r in pool)==expected
    special=[r for r in pool if r['correct']['R6'] and not r['correct']['fixed']];retained=sum(r['correct'][MAIN] for r in special)
    split_gains={s:sum(int(r['correct'][MAIN])-int(r['correct']['R6']) for r in pool if r['split']==s) for s in ['discovery','selection']+(['confirmation'] if phase=='confirmation' else [])}
    gain=spec['gate']['minimum_net_development' if phase=='development' else 'minimum_net_fulltrain']
    criteria=dict(minimum_net=by['R6']['net']>=gain,paired_R6_p=by['R6']['exact_mcnemar_p']<.05,
        strictly_beats_M2_bridge_and_all_own_controls=all(by[k]['net']>0 for k in refs[1:]),
        split_gains_nonnegative=all(v>=0 for v in split_gains.values()),
        retains_at_least_90pct_R6_unique_vs_Fixed=retained>=math.ceil(.9*len(special)),
        all_norm_controls_match=all(r['diagnostics']['norm_match_possible'] for r in pool))
    result=dict(status='complete_cached_development' if phase=='development' else 'complete_cached_fulltrain',phase=phase,n=n,CPU_only=True,GPU_seconds=0,heldout_accessed=False,
        one_fixed_candidate=True,protocol_sha256=sha(LOCK),score_file=path,score_file_sha256=sha(path),passes_gate=all(criteria.values()),criteria=criteria,
        comparisons=comparisons,split_gains=split_gains,retained_R6_special=retained,total_R6_special=len(special),
        next_action='Review actual remaining full-evaluation budget; no automatic model or heldout access' if all(criteria.values()) else 'Stop this recipe; no coefficient, sign, prefix cutoff or reference search',
        limitations='Repeated development, no blind confirmation, no new information or hidden-state causation. R6 additive anchoring does not guarantee correctness retention. No multi-reference claim.')
    with (ROOT/f'results/r14_{phase}_COMPLETE.json').open('x',encoding='utf8') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','confirmation'],required=True);args=p.parse_args();main(args.phase)
