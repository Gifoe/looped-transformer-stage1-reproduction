"""Pure CPU synthetic R7 checks. No real datasets, labels, model or GPU access.

Run: python -B scripts/r7_math_checks.py
Prints JSON evidence to stdout; creates no files or directories.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from r7_math import candidate_scores,token_logits_numpy,token_logits,MODES
from slcd_r6_scores import selected_score


def require(x,message):
    if not x:raise AssertionError(message)


def rejects(fn):
    try:fn()
    except (ValueError,TypeError,RuntimeError):return
    raise AssertionError('Invalid input was accepted')


def logp(x):
    x=x.astype(np.float32);s=x-x.max(axis=-1,keepdims=True)
    return s-np.log(np.exp(s).sum(axis=-1,keepdims=True))


def methods(f,k4,k5=None):
    return {name:dict(normalized_scores=np.asarray(v).tolist(),gold='ignored') for name,v in
            [('loopcd_k4_fixed',f),('native_k4',k4),('native_k5',k4 if k5 is None else k5)]}


def main():
    rng=np.random.default_rng(2021);checks=[]
    mainm=methods([-1.,-1.5,-2.,-1.75],[-1.2,-1.4,-2.3,-1.6],[-1.4,-1.5,-2.,-1.5])
    priorm=methods([-1.1,-1.4,-2.,-1.8],[-1.2,-1.4,-2.2,-1.6])
    single,diag=candidate_scores(mainm,[priorm]);frozen=selected_score(mainm,priorm,'r6_prior_projection')
    require(np.array_equal(single,frozen),'M1 J1 differs from R6')
    repeat,_=candidate_scores(mainm,[priorm]*3);require(np.array_equal(repeat,single),'M1 identical refs differ')
    changed=json.loads(json.dumps(mainm));changed['gold']=99
    for v in changed.values():
        if isinstance(v,dict):v['gold']=-100
    require(np.array_equal(candidate_scores(changed,[priorm])[0],single),'Gold metadata affected score')
    checks+=['M1 exact frozen R6 J1','M1 identical J3 exact','unused gold metadata']
    # Dyadic inputs allow independent additive constants to be represented exactly.
    q4=(rng.integers(-8,9,(4,19))/8).astype(np.float32)
    q1=q4-2*(rng.integers(-8,9,(4,19))/8).astype(np.float32)
    r4=np.repeat(q4[None],3,axis=0)
    r1=r4-2*(rng.integers(-8,9,(3,4,19))/8).astype(np.float32)
    original=[x.copy() for x in (q1,q4,r1,r4)]
    perm=rng.permutation(19);outputs={};diagnostics={}
    for mode in MODES:
        refs=(r1[:1],r4[:1]) if mode=='bridge' else (r1,r4)
        kw=dict(whole_alpha=.375) if mode=='bridge' else dict(permutation=perm) if mode=='shuffle' else {}
        out,d=token_logits_numpy(q1,q4,*refs,mode=mode,**kw);outputs[mode]=out;diagnostics[mode]=d
        zero,_=token_logits_numpy(q1,q4,*refs,mode=mode,force_zero=True,**kw)
        require(np.array_equal(zero,q4+np.float32(.5)*(q4-q1)),'Zero correction differs from Fixed')
        shifted,sd=token_logits_numpy(q1+4,q4-2,refs[0]+8,refs[1]-4,mode=mode,**kw)
        require(np.allclose(d['alpha'],sd['alpha'],atol=2e-14,rtol=2e-14),'Shift changed alpha')
        require(np.allclose(logp(out),logp(shifted),atol=2e-6,rtol=0),'Shift changed probabilities')
    require(all(np.array_equal(a,b) for a,b in zip(original,(q1,q4,r1,r4))),'Input mutation')
    checks+=['all five modes finite','alpha0 exact Fixed all modes','constant-shift invariance within FP32 tolerance','no mutation']
    # All identical references collapse exactly; mixed duplicates retain their weight.
    one,d1=token_logits_numpy(q1,q4,r1[:1],r4[:1])
    three,d3=token_logits_numpy(q1,q4,np.repeat(r1[:1],3,0),np.repeat(r4[:1],3,0))
    require(np.array_equal(one,three) and np.array_equal(d1['alpha'],d3['alpha']),'Identical references differ')
    _,mixed=token_logits_numpy(q1,q4,r1[[0,0,1]],r4[[0,0,1]])
    _,dedup=token_logits_numpy(q1,q4,r1[:2],r4[:2])
    require(not np.array_equal(mixed['robust_denominator'],dedup['robust_denominator']),'Partial duplicates were discarded')
    robust=diagnostics['robust'];matched=diagnostics['matched_scale'];shuffle=diagnostics['shuffle']
    require(np.all(robust['robust_denominator']>=robust['mean_only_denominator']-1e-14),'Jensen denominator failed')
    require(np.any(robust['robust_denominator']>robust['mean_only_denominator']+1e-5),'No denominator control contrast')
    require(np.allclose(matched['rho']**2*matched['guidance_norm2'],robust['residual_norm2'],atol=2e-14,rtol=2e-14),'Matched norm differs')
    require(np.allclose(matched['applied_direction_norm2'],(1-matched['rho'])**2*matched['guidance_norm2'],atol=1e-14),'Matched applied displacement diagnostic differs')
    require(np.allclose(shuffle['applied_direction_norm2'],robust['projected_correction_norm2'],atol=2e-14,rtol=2e-14),'Shuffled norm differs')
    checks+=['all-identical reference exactness','partial duplicate multiplicity retained','robust vs mean denominator','matched guidance norm','shuffled correction norm']
    const=r4-np.float32(2);_,zero=token_logits_numpy(q1,q4,const,r4)
    require(np.all(zero['alpha']==0) and np.all(zero['indistinguishable']),'Constant reference not zeroed')
    # Concentrated p makes a nonconstant direction numerically indistinguishable.
    peaked=np.array([[0.,-1000.,-1000.]],np.float32);ref=np.array([[[0.,1.,-1.]]],np.float32)
    _,underflow=token_logits_numpy(peaked,peaked,np.zeros_like(ref),ref)
    require(np.all(underflow['alpha']==0),'Indistinguishable denominator not rejected')
    zeros=np.zeros_like(q4);_,gzero=token_logits_numpy(zeros,zeros,r1,r4,mode='matched_scale')
    require(np.all(gzero['rho']==1),'Zero guidance norm convention')
    for bad in (np.nan,np.inf):
        broken=q1.copy();broken[0,0]=bad;rejects(lambda:token_logits_numpy(broken,q4,r1,r4))
    rejects(lambda:token_logits_numpy(q1.astype(np.float64),q4,r1,r4))
    rejects(lambda:token_logits_numpy(q1,q4,r1,r4,mode='bridge',whole_alpha=.5))
    rejects(lambda:token_logits_numpy(q1,q4,r1[:1],r4[:1],mode='bridge',whole_alpha=np.nan,force_zero=True))
    rejects(lambda:token_logits_numpy(q1,q4,r1,r4,mode='shuffle',permutation=np.zeros(19,dtype=int)))
    checks+=['zero/indistinguishable denominator','zero guidance rho1','invalid/nonfinite input rejection']
    # Candidate extension cannot alter existing independently computed positions.
    batch,_=token_logits_numpy(np.concatenate([q1,q1[:1]]),np.concatenate([q4,q4[:1]]),
        np.concatenate([r1,r1[:,:1]],axis=1),np.concatenate([r4,r4[:,:1]],axis=1))
    require(np.array_equal(batch[:4],outputs['robust']),'Independent position extension changed logits')
    # logZ prevents linear LL subtraction from reproducing corrected full logits.
    actual=logp(outputs['robust']);naive=logp(q4+np.float32(.5)*(q4-q1))-robust['alpha'][...,None]*logp(r4[0])
    require(not np.allclose(actual,naive,atol=1e-5),'Synthetic log-partition contrast collapsed')
    checks+=['independent position extension','token log-partition is recomputed']
    torch_status='not_installed';maxerror=0.
    try:import torch
    except ImportError:pass
    else:
        torch_status='CPU tested';tq1=torch.tensor(q1,requires_grad=True);tq4=torch.tensor(q4,requires_grad=True)
        for mode in MODES:
            rr=(r1[:1],r4[:1]) if mode=='bridge' else (r1,r4)
            kw=dict(whole_alpha=.375) if mode=='bridge' else dict(permutation=perm) if mode=='shuffle' else {}
            out,d=token_logits(tq1,tq4,torch.tensor(rr[0]),torch.tensor(rr[1]),mode=mode,**kw)
            require(not out.requires_grad and out.grad_fn is None,'Torch output tracks gradients')
            require(all(not v.requires_grad for v in d.values() if isinstance(v,torch.Tensor)),'Diagnostic tracks gradients')
            err=float(np.max(np.abs(out.numpy()-outputs[mode])));maxerror=max(maxerror,err)
            require(np.allclose(out.numpy(),outputs[mode],atol=5e-7,rtol=0),'Torch/NumPy logits disagree')
            for key,value in d.items():
                if isinstance(value,torch.Tensor):require(np.allclose(value.numpy(),diagnostics[mode][key],atol=2e-13,rtol=2e-13),'Torch/NumPy diagnostic differs: '+key)
            zero,_=token_logits(tq1,tq4,torch.tensor(rr[0]),torch.tensor(rr[1]),mode=mode,force_zero=True,**kw)
            require(torch.equal(zero,tq4+.5*(tq4-tq1)),'Torch zero correction not exact')
        require(tq1.grad is None and tq4.grad is None,'Input gradients populated')
        checks+=['Torch CPU independent implementation parity','no grad graph/output','Torch alpha0 exact Fixed']
    script=Path(__file__);mathfile=script.with_name('r7_math.py')
    print(json.dumps(dict(status='passed',scope='CPU synthetic only; no data/model/target-token access',checks=checks,
        torch=torch_status,max_torch_numpy_logit_abs_error=maxerror,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (mathfile,script)},
        numerical_contract='FP32 g/e/Fixed; p=softmax(FP32 z4 converted to FP64); FP64 weighted statistics; final corrected logits cast once to FP32; caller performs FP32 full-vocab log_softmax',
        limits=['No BF16 model/head integration tested here.','Constant-shift invariance is tolerance-bounded because FP32 input/returned logits can round.','Forced-zero returns Fixed; adding the unchanged late term recovers A1, not R6.']),indent=2))


if __name__=='__main__':main()
