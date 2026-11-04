"""Synthetic only math/AUC/paired-bootstrap coverage; no cached research data."""
import numpy as np
from r22_math import numpy_feature,torch_feature
from r22_statistics import evaluate,question_kernel,SIGNALS

def main():
 import torch
 torch.set_num_threads(2)
 def states(h4,h5,h6):return [np.zeros((1,2,3)) for _ in range(3)]+[np.full((1,2,3),v,dtype=np.float64) for v in [h4,h5,h6]]
 for values,want in [((0,0,0),.5),((0,1,1),1.),((0,0,1),0.),((0,1,2),.5)]:
  a=states(*values);x=numpy_feature(a);t=torch_feature([torch.tensor(v,dtype=torch.bfloat16) for v in a]);assert x==t and x['V']==want
 a=states(0,2,3);base=numpy_feature(a)['V'];assert numpy_feature([x+4 for x in a])['V']==base and numpy_feature([x*3 for x in a])['V']==base
 b=states(0,2,3);b[0]=np.zeros((1,0,3))
 try:numpy_feature(b)
 except ValueError:pass
 else:raise AssertionError('Bad shape accepted')
 b=states(0,2,3);b[5][0,0,0]=np.nan
 try:numpy_feature(b)
 except ValueError:pass
 else:raise AssertionError('Nonfinite accepted')
 # Each question weights1: rescue q1 has2pairs[0,2],q2 has1pair[2]; damage1.
 rescue=[np.array([[0.]*5,[2.]*5]),np.array([[2.]*5])];damage=[np.array([[1.]*5])];k=question_kernel(rescue,damage);assert np.all(k.mean((1,2))==.75)
 tie=question_kernel([np.zeros((1,5))],[np.zeros((1,5))]);assert np.all(tie==.5)
 result=evaluate(rescue,damage,repetitions=100);assert result['AUC']['V']==.75 and result['CI95']['V']==[.5,1.] and all(x['CI95']==[0.,0.] for x in result['paired_difference'].values())
 repeated=evaluate(rescue,damage,repetitions=100);assert repeated==result
 return dict(status='passed_CPU_synthetic',stationary_half=True,last_step_zero_one=True,previous_step_zero_zero=True,equal_steps_half=True,shift_scale=True,candidate_extension_by_independent_function=True,invalid_shape_and_nonfinite_rejected=True,NumPy_Torch_exact=True,AUC_tie_half=True,question_weight_AUC_point=.75,paired_bootstrap_identical_signals_zero_difference=True,bootstrap_seed_reproducible=True,scope='Synthetic only; no real cache/features/labels')
if __name__=='__main__':
 import json
 print(json.dumps(main()))

