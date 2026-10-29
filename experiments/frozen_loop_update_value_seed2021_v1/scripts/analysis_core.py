"""Answer-free feature construction and fixed policies, separate from evaluation targets."""
import numpy as np
from scipy.special import softmax,log_softmax
from scipy.stats import binomtest
ACTIONS=['adaptive','keep','forward','rollback']
ALL=ACTIONS+['depth1','depth2','negative']
def view_scores(row,view='official',normalized=True):
 v=row['views'][view];den=np.array(row['choice_char_lengths']) if normalized else 1
 return {k:np.array(v['sum_scores'][k])/den for k in ALL}
def margin(x):
 p=np.sort(softmax(x));return float(p[-1]-p[-2])
def features(row,kind='combined'):
 v=row['views']['official'];s=view_scores(row);c={};n={};b=int(s['adaptive'].argmax())
 c.update(v['controls']);c.update({k:v['geometry'][k] for k in ['norm2','norm3','norm4']})
 for a in ACTIONS:
  lp=log_softmax(s[a]);p=np.exp(lp);c[a+'_margin']=margin(s[a]);c[a+'_entropy']=float(-sum(p*lp));c[a+'_kl_keep']=float(sum(p*(lp-log_softmax(s['keep']))));c[a+'_agrees_adaptive']=float(s[a].argmax()==b)
  n[a+'_probe_delta']=v['probe_nll']['keep']-v['probe_nll'][a]
  # Candidate index never enters a feature; select by observable prediction.
  pred=int(s[a].argmax());ll=np.array(v['token_logprobs'][a][pred]);base=np.array(v['token_logprobs']['keep'][pred]);delta=ll-base
  n[a+'_token_gain_mean']=float(delta.mean());n[a+'_token_positive_fraction']=float((delta>0).mean())
 for a in ['depth1','depth2','rollback','keep']:
  c[a+'_rank_adaptive']=float(np.sum(s[a]>s[a][b]))/max(len(s[a])-1,1)
 for k,val in v['geometry'].items():
  if not k.startswith('norm'):n[k]=val
 for i,val in enumerate(v['update_projections']):n[f'update_projection_{i:03d}']=val
 out=c if kind=='controls' else n if kind=='new' else dict(c,**n)
 assert all(np.isfinite(x) for x in out.values())
 names=sorted(out);return np.array([out[k] for k in names]),names
def fixed_policies(row):
 s=view_scores(row);v=row['views']['official'];s2=view_scores(row,'format')
 choose=lambda fn:max(ACTIONS,key=fn) # Adaptive is first, so ties retain it.
 selected={'proxy':min(ACTIONS,key=lambda a:v['probe_nll'][a]),'geometry':'forward' if v['geometry']['align34']>0 else 'rollback' if v['geometry']['align34']<0 else 'adaptive','confidence':choose(lambda a:margin(s[a])),'support':choose(lambda a:float(np.mean(np.array(v['token_logprobs'][a][int(s[a].argmax())])-np.array(v['token_logprobs']['keep'][int(s[a].argmax())])))),'format_support':choose(lambda a:min(margin(s[a]),margin(s2[a])))}
 pred={a:int(s[a].argmax()) for a in ALL}
 pred.update({k:int(s[a].argmax()) for k,a in selected.items()})
 pred['ensemble']=int(((s['adaptive']+s2['adaptive'])/2).argmax())
 return pred,selected
def targets(row,kind='accuracy'):
 s=view_scores(row);gold=row['gold']
 if kind=='accuracy':q={a:float(s[a].argmax()==gold) for a in ACTIONS}
 else:q={a:float(log_softmax(s[a])[gold]) for a in ACTIONS}
 return np.array([q[a]-q['adaptive'] for a in ACTIONS[1:]])
def paired(y,base,resamples=10000):
 d=np.asarray(y,dtype=int)-np.asarray(base,dtype=int);n=len(d);plus=int((d>0).sum());minus=int((d<0).sum())
 rng=np.random.default_rng(2021)
 # Exact paired empirical bootstrap via multinomial counts of -1,0,+1.
 counts=rng.multinomial(n,[minus/n,(n-minus-plus)/n,plus/n],size=resamples);samples=(counts[:,2]-counts[:,0])/n*100
 return dict(n=n,correct=int(np.sum(y)),accuracy_percent=100*float(np.mean(y)),delta_pp=100*float(d.mean()),corrections=plus,degradations=minus,net=plus-minus,ci95_pp=np.quantile(samples,[.025,.975]).tolist(),mcnemar_p=float(binomtest(plus,plus+minus).pvalue) if plus+minus else 1.)
def holm(p):
 keys=sorted(p,key=p.get);out={};prev=0
 for i,k in enumerate(keys):prev=max(prev,min(1.,p[k]*(len(keys)-i)));out[k]=prev
 return out
