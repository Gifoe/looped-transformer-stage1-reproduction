from fractions import Fraction
import torch
from ab1_metrics import score,gaps,fraction_gap,fixedwrong,edges
f={'H00':0,'H10':2,'H01':3,'H11':9};assert edges(f)=={'Q_R_E':2,'Q_R_B':6,'R_Q_E':3,'R_Q_B':7,'interaction':4};assert edges({k:Fraction(v) for k,v in f.items()})['interaction']==Fraction(4)
v=[1.,2.,2.,0.];assert fixedwrong(v,0)==1 and gaps(v,0,1)['dynamic']==-1.;assert fraction_gap([1.,2.,2.,0.],0,1)['allwrong']==Fraction(-1,3)
z=torch.zeros(49152,dtype=torch.bfloat16);z[340]=z[330]=2;s=score(z,[340,330,389]);assert s['prediction']==0 and s['ties']==[0,1] and s['candidate_logits']==[2.,2.,0.]
print('PASS AB1 edge algebra, candidate gold-vs-allwrong exact fractions, fixedwrong first-tie, full-vocab BF16 scoring; synthetic only')
