import numpy as np
from e3_value_partition import split_value_interaction
rng=np.random.default_rng(902);a=rng.random((1,3,8));b=rng.random((1,3,8));a/=a.sum(-1,keepdims=True);b/=b.sum(-1,keepdims=True);v=rng.normal(size=(1,3,8,5));w=rng.normal(size=v.shape);out,total=split_value_interaction(a,b,v,w,7,4);assert np.allclose(out['P']+out['Q']+out['R'],total);assert np.allclose(out['R'],(b-a)[:,:,7,None]*(w-v)[:,:,7,:]);
try:split_value_interaction(a,b,v,w,3,4)
except ValueError:pass
else:raise AssertionError('R crossed prefix silently')
print('E3_P_Q_R_INTERACTION_ACCOUNTING_CPU_PASS')
