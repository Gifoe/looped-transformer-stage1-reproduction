"""FP64 lens and observed BF16 residual-addition error; no causal claim."""
import numpy as np

def decompose(states,W,gamma,eps,targets,gold):
 W=np.asarray(W,dtype=np.float64);gamma=np.asarray(gamma,dtype=np.float64)
 if len(targets)<2 or gold not in range(len(targets)) or eps<=0:raise ValueError('labels/eps')
 out=[]
 for u in (0,1):
  for i in range(24):
   d=states[(u,i)];v={k:d[k].detach().cpu().float().numpy()[0].astype(np.float64) for k in ('x','a','y','b','z')}
   if any(not np.isfinite(a).all() for a in v.values()):raise ValueError('nonfinite')
   norms={k:float(np.sqrt(np.mean(v[k]**2)+eps)) for k in ('x','y','z')}
   z={k:W[targets]@(gamma*v[k]/norms[k]) for k in ('x','y','z')}
   def step(left,right):
    sx=norms[left];sy=norms[right];nx=z[left]*sx;ny=z[right]*sy
    numerator=.5*(ny-nx)*(1/sx+1/sy);denominator=.5*(nx+ny)*(1/sy-1/sx);change=z[right]-z[left]
    return {'delta':change.tolist(),'numerator':numerator.tolist(),'denominator':denominator.tolist(),'identity_max_error':float(np.max(np.abs(numerator+denominator-change)))}
   out.append({'loop':u+1,'layer':i+1,'attention':step('x','y'),'mlp':step('y','z'),'attention_bf16_addition_residual':(v['y']-v['x']-v['a']).tolist(),'mlp_bf16_addition_residual':(v['z']-v['y']-v['b']).tolist(),'gold':gold})
 return out
