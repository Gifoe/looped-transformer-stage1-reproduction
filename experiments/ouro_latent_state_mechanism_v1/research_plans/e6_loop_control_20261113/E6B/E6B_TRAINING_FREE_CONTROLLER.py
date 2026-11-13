"""Frozen small E6B controller family. Only observables enter predict()."""
import numpy as np

def probability(s):
 s=np.asarray(s,dtype=float);p=np.exp(s-s.max());return p/p.sum()
def output_features(scores,depth):
 s=np.asarray(scores,dtype=float);p=probability(s[depth-1]);prev=probability(s[max(depth-2,0)]);order=np.argsort(s[depth-1]);last=np.argsort(s[max(depth-2,0)]);m=(p+prev)/2
 js=.5*np.sum(p*np.log(np.maximum(p,1e-30)/np.maximum(m,1e-30)))+.5*np.sum(prev*np.log(np.maximum(prev,1e-30)/np.maximum(m,1e-30)))
 agreement=float(int(np.argmax(p))==int(np.argmax(prev)));streak=1
 for d in range(depth-2,-1,-1):
  if np.argmax(s[d])==np.argmax(p):streak+=1
  else:break
 return np.array([-np.sum(p*np.log(np.maximum(p,1e-30)))/np.log(len(p)),s[depth-1,order[-1]]-s[depth-1,order[-2]],p.max(),agreement,js,float(np.mean(order==last)),float(s[depth-1].std()),len(p),depth,streak],float)

def threshold_from_unlabeled_train(rows,quantile=.5):
 return {str(d):float(np.quantile([r['latent'][d-1][11] for r in rows if not r.get('is_demonstration',False)],quantile)) for d in [2,3,4]}

def early_exit_prefix(scores_prefix,latent_prefix,thresholds,use_latent=True,p_threshold=.9):
 d=len(scores_prefix);assert d in [2,3] and len(latent_prefix)==d
 f=output_features(scores_prefix,d)
 return bool(f[3] and f[2]>=p_threshold and (not use_latent or latent_prefix[d-1][11]<=thresholds[str(d)]))

def predict(scores,latent,thresholds,strategy='rollback',use_latent=True,p_threshold=.9):
 scores=np.asarray(scores,float);pred=[int(np.argmax(s)) for s in scores];sel=3;executed=4
 if strategy=='rollback':
  cf3=output_features(scores,3);cf4=output_features(scores,4)
  if pred[2]!=pred[3] and cf3[2]>cf4[2] and (not use_latent or latent[3][11]>thresholds['4']):sel=2
 elif strategy=='agreement':
  cf=output_features(scores,4)
  if pred[2]!=pred[3] and cf[0]>.5 and (not use_latent or latent[3][5]<latent[2][5]):sel=2
 elif strategy in ['stability','early_exit']:
  for d in [2,3]:
   f=output_features(scores,d)
   if early_exit_prefix(scores[:d],latent[:d],thresholds,use_latent,p_threshold):
    sel=d-1
    if strategy=='early_exit':executed=d
    break
 else:raise ValueError(strategy)
 return {'prediction':pred[sel],'selected_depth':sel+1,'executed_loops':executed,'score_protocol':'joint_option_letter','uses_gold':False}
