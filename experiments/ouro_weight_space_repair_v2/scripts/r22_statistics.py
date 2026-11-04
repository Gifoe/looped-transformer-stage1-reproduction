"""Question-weighted AUC, paired stratified question bootstrap; no fitting."""
import numpy as np
SIGNALS=['V','C_energy','C_margin','C_log_tokens','C_neg_log_tokens']

def question_kernel(rescue,damage):
 # Each input question is [pairs,signals]; each question contributes total1.
 return np.asarray([[[np.mean((a[:,k,None]>b[None,:,k])+.5*(a[:,k,None]==b[None,:,k])) for b in damage] for a in rescue] for k in range(len(SIGNALS))],dtype=np.float64)

def evaluate(rescue,damage,seed=2021,repetitions=10000):
 if not rescue or not damage:return dict(status='no_two_classes',rescue_questions=len(rescue),damage_questions=len(damage))
 K=question_kernel(rescue,damage);auc=K.mean(axis=(1,2));rng=np.random.default_rng(seed);nr=len(rescue);nd=len(damage);boot=[]
 for begin in range(0,repetitions,100):
  count=min(100,repetitions-begin);ir=rng.integers(nr,size=(count,nr));id=rng.integers(nd,size=(count,nd));wr=np.stack([np.bincount(x,minlength=nr)/nr for x in ir]);wd=np.stack([np.bincount(x,minlength=nd)/nd for x in id]);boot.append(np.stack([np.sum((wr@k)*wd,axis=1) for k in K],axis=1))
 boot=np.concatenate(boot);cis=np.quantile(boot,[.025,.975],axis=0);diff=boot[:,0,None]-boot[:,1:]
 return dict(status='complete',rescue_questions=nr,damage_questions=nd,rescue_pairs=sum(len(x) for x in rescue),damage_pairs=sum(len(x) for x in damage),AUC={k:float(v) for k,v in zip(SIGNALS,auc)},CI95={k:cis[:,i].tolist() for i,k in enumerate(SIGNALS)},paired_difference={k:dict(point=float(auc[0]-auc[i]),CI95=np.quantile(diff[:,i-1],[.025,.975]).tolist()) for i,k in enumerate(SIGNALS) if i},bootstrap_repetitions=repetitions,seed=seed,question_weight_each_class=1,paired_same_indices_all_signals=True)
