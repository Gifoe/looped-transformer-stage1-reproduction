"""Pure post-forward AB1 score and question-wise analysis; no selection or model call."""
from fractions import Fraction
import torch
if not __debug__:raise RuntimeError('optimized Python forbidden')
def score(logits,targets):
 z=logits.reshape(-1);assert z.numel()==49152 and z.dtype==torch.bfloat16 and torch.isfinite(z).all() and len(targets)>=2 and len(set(targets))==len(targets) and all(0<=t<z.numel() for t in targets)
 v=z[targets];lp=z.float().log_softmax(-1);ties=torch.nonzero(v==v.max()).reshape(-1).tolist();return {'candidate_logits':[float(x) for x in v], 'scores':[float(x) for x in lp[targets]],'prediction':ties[0],'ties':ties,'logsumexp':float(z.float().logsumexp(-1))}
def gaps(v,g,fw):
 assert len(v)>=2 and 0<=g<len(v) and 0<=fw<len(v) and fw!=g
 return {'allwrong':v[g]-sum((x for i,x in enumerate(v) if i!=g),v[0]*0)/(len(v)-1),'fixedwrong':v[g]-v[fw],'dynamic':v[g]-max(x for i,x in enumerate(v) if i!=g)}
def fraction_gap(logits,g,fw):return gaps([Fraction(x) for x in logits],g,fw)
def fixedwrong(scores,g):
 assert len(scores)>=2 and 0<=g<len(scores)
 return max((i for i in range(len(scores)) if i!=g),key=lambda i:scores[i])
def edges(f):
 assert set(('H00','H10','H01','H11'))<=set(f)
 return {'Q_R_E':f['H10']-f['H00'],'Q_R_B':f['H11']-f['H01'],'R_Q_E':f['H01']-f['H00'],'R_Q_B':f['H11']-f['H10'],'interaction':f['H11']-f['H10']-f['H01']+f['H00']}
