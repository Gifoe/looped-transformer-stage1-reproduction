"""Label-free inference primitives. Gold labels never enter this module."""
import numpy as np

def probabilities(x):
    z=np.asarray(x,dtype=np.float64);z=z-z.max(axis=-1,keepdims=True)
    p=np.exp(z);return p/p.sum(axis=-1,keepdims=True)

def infer(scores,spec):
    x=np.asarray(scores,dtype=np.float64)
    assert x.ndim==2 and x.shape[0]==4 and 3<=x.shape[1]<=5 and np.isfinite(x).all()
    x=x-x.mean(axis=1,keepdims=True);p=probabilities(x);pred=x.argmax(axis=1)
    margin=np.sort(p,axis=1)[:,-1]-np.sort(p,axis=1)[:,-2]
    ent=-(p*np.log(np.maximum(p,1e-300))).sum(axis=1)/np.log(x.shape[1])
    kind=spec['kind'];depth=4;v=x[3].copy()
    if kind=='native':depth=spec['depth'];v=x[depth-1]
    elif kind in ['score','probability','log_probability','rank']:
        w=np.asarray(spec['weights'],dtype=float);assert (w>=0).all() and w.sum()>0;w=w/w.sum()
        if kind=='score':a=x
        elif kind=='probability':a=p
        elif kind=='log_probability':a=np.log(np.maximum(p,1e-300))
        else:a=-np.asarray([[sum(b>a for b in row)+.5*(sum(b==a for b in row)-1) for a in row] for row in x])
        v=w@a
    elif kind=='confidence':
        q=margin if spec['signal']=='margin' else -ent;depth=int(np.flatnonzero(q[1:]==q[1:].max())[-1])+2;v=x[depth-1]
    elif kind=='vote':
        count=np.bincount(pred[1:],minlength=x.shape[1]);winners=np.flatnonzero(count==count.max());choice=pred[3] if pred[3] in winners else int(winners[0]);v=np.zeros(x.shape[1]);v[choice]=1
    elif kind=='stability':
        depth=3 if pred[1]==pred[2] else 4;v=x[depth-1]
    elif kind=='conservative':
        depth=3 if pred[0]==pred[1]==pred[2] and pred[3]!=pred[2] and margin[3]<margin[2] else 4;v=x[depth-1]
    elif kind=='rollback':
        depth=3 if pred[2]!=pred[3] and margin[2]>=2*margin[3] and p[2].max()>=.8 else 4;v=x[depth-1]
    elif kind=='early':
        # Prefix-only: branch k cannot access p[k+1:] or any future state.
        for k in (1,2):
            if pred[k]==pred[k-1] and p[k].max()>=spec['threshold'] and ent[k]<=spec['entropy']:
                depth=k+1;break
        v=x[depth-1]
    else:raise ValueError(kind)
    return {'prediction':int(np.argmax(v)),'selected_depth':depth,'executed_loops':depth if kind in ['early','native'] else 4,'scores':v.tolist(),'top_tie':bool(sum(v==max(v))>1)}

def fixed_specs():
    out={f'K{k}':{'kind':'native','depth':k,'track':'ZL'} for k in range(1,5)}
    for name,w in [('uniform34',[0,0,1,1]),('uniform234',[0,1,1,1]),('uniform1234',[1,1,1,1]),('weighted4',[0,1,1,2])]:out[name]={'kind':'score','weights':w,'track':'ZL'}
    for kind in ['probability','log_probability','rank']:out[kind]={'kind':kind,'weights':[0,1,1,1],'track':'ZL'}
    for sig in ['margin','entropy']:out['highest_'+sig]={'kind':'confidence','signal':sig,'track':'ZL'}
    for kind in ['vote','stability','conservative','rollback']:out[kind]={'kind':kind,'track':'ZL'}
    out['early_fixed']={'kind':'early','threshold':.9,'entropy':.3,'track':'ZL'}
    return out

def calibration_grid():
    out={}
    for w in [(0,0,1,2),(0,0,1,3),(0,1,1,3),(0,1,2,4),(1,1,1,4),(1,1,2,4),(0,1,2,2),(0,1,0,1)]:out['cal_weight_'+''.join(map(str,w))]={'kind':'score','weights':list(w),'track':'Cal'}
    for t in [.6,.7,.8,.9,.95,.99]:out['cal_early_'+str(t)]={'kind':'early','threshold':t,'entropy':.5,'track':'Cal'}
    return out
