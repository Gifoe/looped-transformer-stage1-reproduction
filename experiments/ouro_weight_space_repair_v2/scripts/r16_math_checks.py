"""Synthetic-only solver checks; grids here verify convexity, never tune on data."""
import json
import numpy as np
from r16_math import numpy_logits,torch_logits,_logp,STEPS


def main():
    import torch
    rng=np.random.default_rng(2021);q=(rng.integers(-12,13,(7,19))/8).astype('float32');q1=q+(rng.integers(-8,9,q.shape)/8).astype('float32')
    r4=(rng.integers(-15,16,(3,)+q.shape)/8).astype('float32');r1=r4+(rng.integers(-8,9,r4.shape)/8).astype('float32');perm=rng.permutation(q.shape[-1])
    z,d=numpy_logits(q1,q,r1,r4);controls,actual=torch_logits(*(torch.from_numpy(x) for x in [q1,q,r1,r4]),permutation=perm)
    error=float(np.max(np.abs(d['alpha']-actual['alpha'].numpy())));assert error<=2**(-STEPS)
    assert np.max(np.abs(z-controls['exact_native_KL'].numpy()))<2e-6
    for name in ['alpha','dispersion','derivative0','derivative1','objective_ideal','objective_FP32','rho']:
        assert np.max(np.abs(d[name]-actual[name].numpy()))<1e-9
    assert bool((actual['alpha_interval_width']<=2**(-STEPS)).all())
    assert np.max(np.abs(actual['matched_guidance_norm2'].numpy()-actual['residual_norm2'].numpy()))<1e-11
    assert np.max(np.abs(actual['shuffled_correction_norm2'].numpy()-actual['correction_norm2'].numpy()))<1e-11
    # Independent dense-grid oracle for synthetic logits only.
    p=np.exp(_logp(q));e=.5*(r4-r1);ec=e.astype(float)-e[...,:1];ec-=np.sum(p[None]*ec,axis=-1,keepdims=True);bar=ec.mean(0);U=np.mean(np.sum(p[None]*(ec-bar)**2,axis=-1),axis=0)
    fixed=q+.5*(q-q1);lpF=_logp(fixed)
    for a in np.linspace(0,1,257):
        obj=np.sum(p*(lpF-_logp(fixed.astype(float)-a*bar)),axis=-1)+.5*a*a*U
        assert np.all(d['objective_ideal']<=obj+1e-11)
    # No direction and no guidance return literal original Fixed.
    no_e,ne=numpy_logits(q1,q,r4,r4);assert np.array_equal(no_e,fixed) and np.all(ne['alpha']==0)
    no_g,ng=numpy_logits(q,q,r1,r4);assert np.array_equal(no_g,q) and np.all(ng['alpha']==0)
    zero,_=torch_logits(*(torch.from_numpy(x) for x in [q1,q,r1,r4]),permutation=perm,force_zero=True);assert torch.equal(zero['exact_native_KL'],torch.from_numpy(fixed))
    # Full collection repetition preserves weights; identical refs collapse exactly.
    zd,dd=numpy_logits(q1,q,np.tile(r1,(2,1,1)),np.tile(r4,(2,1,1)));assert np.max(np.abs(dd['alpha']-d['alpha']))<=2**(-STEPS)
    one,od=numpy_logits(q1,q,r1[:1],r4[:1]);dup,dup_d=numpy_logits(q1,q,np.repeat(r1[:1],3,0),np.repeat(r4[:1],3,0));assert np.array_equal(one,dup) and np.array_equal(od['alpha'],dup_d['alpha'])
    shifted,sd=numpy_logits(q1+4,q+4,r1+2,r4+2);assert np.max(np.abs(sd['alpha']-d['alpha']))<=2**(-STEPS)
    assert np.max(np.abs(np.exp(_logp(shifted))-np.exp(_logp(z))))<2e-6
    # Curvature can activate a correction absent in the Fisher approximation.
    qq=np.zeros((1,4),dtype='float32');g=np.array([[2,1,-1,-2]],dtype='float32');ee=np.array([[1,-1,-1,1]],dtype='float32')
    nz,nd=numpy_logits(qq-2*g,qq,(-2*ee)[None],qq[None]);assert nd['alpha'][0]>0 and nd['rho'][0]>1 and float(np.sum(g*ee))==0
    tc,td=torch_logits(*(torch.from_numpy(x) for x in [qq-2*g,qq,(-2*ee)[None],qq[None]]),permutation=np.arange(4));assert td['rho'].item()>1 and td['old_Fisher_alpha'].item()==0
    assert abs(td['matched_guidance_norm2'].item()-td['residual_norm2'].item())<1e-12
    result=dict(status='passed_synthetic_solver_checks',NumPy_Torch_max_alpha_error=error,bisection_steps=STEPS,synthetic_grid_only=True,
        zero_repair_exact_Fixed=True,flat_direction_exact_Fixed=True,no_guidance_exact_native=True,duplicate_reference_exact=True,
        objective_no_worse_than_synthetic_grid=True,unclipped_rho_control_checked=True,
        nonlinear_counterexample=dict(old_Fisher_alpha=0.,new_alpha=float(nd['alpha'][0]),rho=float(nd['rho'][0])),
        interpretation='Standard native-distribution KL calibration; no correctness guarantee, no empirical dataset outcomes used.')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':main()
