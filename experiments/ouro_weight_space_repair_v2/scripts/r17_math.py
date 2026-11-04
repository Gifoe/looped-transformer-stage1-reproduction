"""Paired answer-cue intervention responses in a shared native-Q4 metric."""
import numpy as np
EPS=64.*np.finfo(np.float64).eps


def numpy_stats(q1,q4,n1,n4,sq1,sq4,sn1,sn4,permutation):
    values=[np.asarray(x) for x in [q1,q4,n1,n4,sq1,sq4,sn1,sn4]]
    assert all(x.shape==q4.shape and x.dtype==np.float32 and np.isfinite(x).all() for x in values)
    def prob(z):
        z=z.astype(float);v=np.exp(z-z.max(-1,keepdims=True));return v/v.sum(-1,keepdims=True)
    p=prob(q4);pn=prob(n4);g=.5*(q4-q1);e=(.5*(n4-n1)).astype(float)
    vn=(.5*(sn4-sn1)).astype(float)-e;vq=(.5*(sq4-sq1)).astype(float)-g.astype(float)
    def center(x,w):
        shifted=x-x[:,:1];return shifted-np.sum(w*shifted,axis=-1,keepdims=True)
    def pair(a,b,w):
        a,b=center(a,w),center(b,w);a2=np.sum(w*a*a,axis=-1);b2=np.sum(w*b*b,axis=-1);inner=np.sum(w*a*b,axis=-1)
        valid=(a2>EPS*np.max(abs(a),axis=-1)**2)&(b2>EPS*np.max(abs(b),axis=-1)**2)
        cos=np.divide(inner,np.sqrt(a2*b2),out=np.zeros_like(inner),where=valid)
        return dict(cosine=cos,R2=cos*cos,defined=valid,a_norm2=a2,b_norm2=b2,inner=inner)
    output={}
    for name,a,b,w in [('reference',e,vn,p),('transfer',vn,vq,p),('reference_shuffle',e[:,permutation],vn,p),('transfer_shuffle',vn[:,permutation],vq,p),('reference_native_metric',e,vn,pn),('static',e,g.astype(float),p)]:
        for k,v in pair(a,b,w).items():output[name+'_'+k]=v
    total=output['transfer_a_norm2']+output['transfer_b_norm2'];valid=total>0
    output['identity_error']=np.divide(total-2*output['transfer_inner'],total,out=np.zeros_like(total),where=valid);output['identity_error_defined']=valid
    output['slope']=np.divide(output['transfer_inner'],output['transfer_a_norm2'],out=np.zeros_like(total),where=output['transfer_a_norm2']>0)
    return output


def torch_stats(q1,q4,n1,n4,sq1,sq4,sn1,sn4,permutation):
    import torch
    values=[q1,q4,n1,n4,sq1,sq4,sn1,sn4];assert all(x.shape==q4.shape and x.dtype==torch.float32 and bool(torch.isfinite(x).all()) for x in values)
    p=q4.double().softmax(-1);pn=n4.double().softmax(-1);g=.5*(q4-q1);e=(.5*(n4-n1)).double()
    vn=(.5*(sn4-sn1)).double()-e;vq=(.5*(sq4-sq1)).double()-g.double();perm=torch.as_tensor(permutation,device=q4.device)
    assert perm.shape==(q4.shape[-1],) and torch.equal(perm.sort().values,torch.arange(q4.shape[-1],device=q4.device))
    def center(x,w):
        shifted=x-x[:,:1];return shifted-(w*shifted).sum(-1,keepdim=True)
    def pair(a,b,w):
        a,b=center(a,w),center(b,w);a2=(w*a*a).sum(-1);b2=(w*b*b).sum(-1);inner=(w*a*b).sum(-1)
        valid=(a2>EPS*a.abs().amax(-1).square())&(b2>EPS*b.abs().amax(-1).square());den=torch.sqrt(a2*b2)
        cos=torch.where(valid,inner/torch.where(valid,den,torch.ones_like(den)),torch.zeros_like(den))
        assert bool((cos.abs()<=1+1e-12).all())
        return dict(cosine=cos,R2=cos.square(),defined=valid,a_norm2=a2,b_norm2=b2,inner=inner)
    output={}
    for name,a,b,w in [('reference',e,vn,p),('transfer',vn,vq,p),('reference_shuffle',e[:,perm],vn,p),('transfer_shuffle',vn[:,perm],vq,p),('reference_native_metric',e,vn,pn),('static',e,g.double(),p)]:
        for k,v in pair(a,b,w).items():output[name+'_'+k]=v
    total=output['transfer_a_norm2']+output['transfer_b_norm2'];valid=total>0
    output['identity_error']=torch.where(valid,(total-2*output['transfer_inner'])/torch.where(valid,total,torch.ones_like(total)),torch.zeros_like(total));output['identity_error_defined']=valid
    output['slope']=torch.where(output['transfer_a_norm2']>0,output['transfer_inner']/torch.where(output['transfer_a_norm2']>0,output['transfer_a_norm2'],torch.ones_like(total)),torch.zeros_like(total))
    assert bool(((output['identity_error']>=-1e-12)&(output['identity_error']<=2+1e-12)).all())
    return output


def synthetic_checks():
    import torch
    rng=np.random.default_rng(2021);v=[(rng.integers(-24,25,(5,13))/8).astype('float32') for _ in range(8)];perm=rng.permutation(13)
    a=numpy_stats(*v,perm);b=torch_stats(*(torch.from_numpy(x) for x in v),permutation=perm);error=0.
    for k in a:
        if a[k].dtype==bool:assert np.array_equal(a[k],b[k].numpy())
        else:error=max(error,float(np.max(np.abs(a[k]-b[k].numpy()))))
    assert error<1e-11
    zero=numpy_stats(*v[:4],*v[:4],perm);assert not zero['reference_defined'].any() and not zero['transfer_defined'].any() and not zero['identity_error_defined'].any()
    # Explicit synthetic identity and orthogonal-response witnesses, no data.
    z=np.zeros((1,4),dtype='float32');e=np.array([[1,-1,1,-1]],dtype='float32');u=np.array([[1,1,-1,-1]],dtype='float32')
    same=numpy_stats(z,z,-2*e,z,-2*e,z,-4*e,z,np.arange(4));assert same['reference_R2'][0]==1 and same['transfer_R2'][0]==1 and same['identity_error'][0]==0
    orth=numpy_stats(z,z,-2*e,z,-2*u,z,-4*e,z,np.arange(4));assert orth['reference_R2'][0]==1 and orth['transfer_R2'][0]==0 and orth['identity_error'][0]==1
    gauge=numpy_stats(*(x+4 for x in v),permutation=perm)
    for k in a:
        if a[k].dtype!=bool:assert np.max(np.abs(a[k]-gauge[k]))<1e-11
    return dict(status='passed_synthetic_paired_response_checks',NumPy_Torch_max_error=error,zero_response_undefined=True,identity_and_orthogonal_witnesses=True,logit_gauge_checked=True,
        interpretation='An actual cue intervention response, not a correctness or pure-nuisance ground truth.')


if __name__=='__main__':
    import json
    print(json.dumps(synthetic_checks(),indent=2))
