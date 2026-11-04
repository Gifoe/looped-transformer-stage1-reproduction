"""Finite development-only sign falsification. No model or heldout path exists.

Known cyclic contrast/fusion operations; no novelty or deployment claim.
Definitions are fixed before this diagnostic's outcomes, after earlier rounds.
"""
import csv,hashlib,json
import numpy as np
from r7_core import ROOT,read,sha,jsonlines,digest
from r8_scores import center,norm,EPS,finite
from slcd_r6_scores import _inputs,selected_score

LOCK='configs/r12_SIGNED_DIAGNOSTIC_LOCK.json'


def score(a,r,e,references):
    # Pure inference function: no labels, IDs, texts or fitted parameters.
    j4=np.asarray([x['sum_scores']['native_k4'] for x in references],dtype=np.float64)
    j1=np.asarray([x['sum_scores']['native_k1'] for x in references],dtype=np.float64)
    finite(a,r,e,j4,j1)
    s=center((j4-j1).mean(axis=0));static=center(j4.mean(axis=0));size=norm(center(r))
    sn=norm(s);un=norm(static);scale=max(norm(center(v)) for v in list(j4)+list(j1))
    delta=np.zeros_like(r) if sn<=EPS*scale else size*s/sn
    u=np.zeros_like(r) if un<=EPS*max(norm(j4.mean(axis=0)),1.) else size*static/un
    positive=r+delta;target=norm(center(positive-a));en=norm(center(e))
    possible=en>0. or target==0.
    matched=a.copy() if en==0. else a-target*e/en
    values=dict(signed_loop=positive,reversed_loop=r-delta,static_joint=r+u,prior_norm=matched,
                cyclic_joint4_only=j4.mean(axis=0),R6=r,A1=a)
    finite(*values.values())
    return values,dict(zero_loop_axis=bool(sn<=EPS*scale),prior_norm_match_possible=possible,
        correction_norm=target,matched_norm=norm(center(matched-a)),increment_norm=norm(delta))


def main():
    lock=read(LOCK);assert sha('scripts/r12_signed_diagnostic.py')==lock['source_sha256']
    assert lock['heldout_access_authorized'] is False and lock['GPU_seconds']==0
    for p,h in lock['input_sha256'].items():assert sha(p)==h
    receipt=read('results/r10_development_initial_COMPLETE.json');assert receipt['status']=='complete' and receipt['n']==576 and sha(receipt['score_file'])==receipt['score_file_sha256']
    refs={}
    for split in ['discovery','selection']:
        for row in jsonlines('results/r7_'+split+'_development.jsonl'):
            refs[row['id']]=row
    aux=list(jsonlines(receipt['score_file']));assert len(refs)==len(aux)==len({r['id'] for r in aux})==576 and set(refs)=={r['id'] for r in aux}
    out=[]
    for row in aux:
        ref=refs[row['id']];assert row['signature']==receipt['signature'] and row['gold']==ref['gold'] and row['prompt_sha256']==ref['prompt_sha256']
        a,g,e=_inputs(ref['original_methods'],ref['prior_methods'][0]);r=np.asarray(ref['baseline_scores']['M0_R6'])
        assert np.array_equal(r,selected_score(ref['original_methods'],ref['prior_methods'][0],'r6_prior_projection'))
        values,diag=score(a,r,e,row['cyclic_rotations']);pred={k:int(np.argmax(v)) for k,v in values.items()}
        out.append(dict(id=row['id'],split=row['split'],gold=row['gold'],scores={k:v.tolist() for k,v in values.items()},predictions=pred,correct={k:v==row['gold'] for k,v in pred.items()},diagnostics=diag,lock_sha256=sha(LOCK)))
    assert sum(r['correct']['R6'] for r in out)==360
    result=[]
    for name in out[0]['scores']:
        result.append(dict(method=name,n=576,correct=sum(r['correct'][name] for r in out),rescue=sum(r['correct'][name] and not r['correct']['R6'] for r in out),damage=sum(not r['correct'][name] and r['correct']['R6'] for r in out)))
    counts={r['method']:r['correct'] for r in result};match=all(r['diagnostics']['prior_norm_match_possible'] for r in out)
    gate=match and all(counts['signed_loop']>counts[k] for k in ['R6','reversed_loop','static_joint','prior_norm'])
    bothwrong=[r for r in out if not r['correct']['R6'] and not r['correct']['cyclic_joint4_only']]
    # Always stop here: this is one falsification, never an auto-promotion.
    receipt_out=dict(status='completed_development_falsification',n=576,CPU_only=True,GPU_seconds=0,heldout_accessed=False,lock_sha256=sha(LOCK),summary=result,
        passes_all_diagnostic_controls=gate,prior_norm_match_possible_all=match,
        both_R6_and_joint_wrong_n=len(bothwrong),signed_rescues_when_both_wrong=sum(r['correct']['signed_loop'] for r in bothwrong),
        signed_damages_joint_correct=sum(not r['correct']['signed_loop'] and r['correct']['cyclic_joint4_only'] for r in out),
        signed_beats_joint_only=counts['signed_loop']>counts['cyclic_joint4_only'],
        next_action='Stop; no validation, confirmation or test access authorized by this diagnostic.',
        interpretation='Repeated development only. Sign evidence is not novelty, a causal hidden-state mechanism, or proof of useful LoopCD deployment.')
    path=ROOT/'results/r12_signed_development.jsonl';assert not path.exists()
    with path.open('x',encoding='utf8') as f:
        for row in out:f.write(json.dumps(row,allow_nan=False)+'\n')
    with (ROOT/'results/r12_signed_summary.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    receipt_out['score_file_sha256']=sha('results/r12_signed_development.jsonl')
    with (ROOT/'results/r12_SIGNED_DIAGNOSTIC_COMPLETE.json').open('x',encoding='utf8') as f:f.write(json.dumps(receipt_out,indent=2)+'\n')
    print(json.dumps(receipt_out),flush=True)


if __name__=='__main__':main()
