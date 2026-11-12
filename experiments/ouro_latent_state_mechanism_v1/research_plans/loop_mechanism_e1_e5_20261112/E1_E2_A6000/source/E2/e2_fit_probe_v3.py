"""Fit fixed masked linear candidate probes on train; choose penalties on validation; never evaluate test."""
import os,json,gzip,hashlib,datetime
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from safetensors import safe_open
R=Path('/home/luansiyu/loop-latent');B=R/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E2';ST=E/'E2_fixed_probe_states.safetensors';ROWS=E/'E2_latent_transition_rows.jsonl.gz';AUDIT=E/'E2_posthoc_audit.json';ORDER=B/'orders/E2_PROBE_FIT_ORDER_v3.md';LAMBDA=[1e-4,1e-2,1.,1e2,1e4];SEED=20261112
if not __debug__:raise RuntimeError('optimized Python forbidden')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
audit=json.loads(AUDIT.read_text());assert audit['status']=='E2_POSTHOC_SOURCE_AND_ALL_PROBE_STATES_AUDIT_PASSED' and audit['count']==2590 and sha(ST)==audit['source_sha256'][str(ST)] and sha(ROWS)==audit['source_sha256'][str(ROWS)]
with gzip.open(ROWS,'rt') as f:data=[r for line in f if (r:=json.loads(line))['split'] in ('train','validation')]
train=[r for r in data if r['split']=='train' and not r['demo']]; val=[r for r in data if r['split']=='validation'];assert len(train)==1094 and len(val)==299 and sum(r['demo'] for r in data if r['split']=='train')==25
classes=sorted({t for r in train for t in r['targets']});ci={x:i for i,x in enumerate(classes)};C=len(classes);assert C>=2

def fields(rr):
 mask=np.array([[x in z['targets'] for x in classes] for z in rr],dtype=bool); y=np.array([ci.get(z['targets'][z['gold']],-1) for z in rr]);assert mask.any(axis=1).all() and all(y[i]<0 or mask[i,y[i]] for i in range(len(rr)));return mask,y
mt,yt=fields(train);mv,yv=fields(val);assert (yt>=0).all();val_supported=np.array([all(t in ci for t in z['targets']) and yv[i]>=0 for i,z in enumerate(val)]); yv=np.where(val_supported,yv,-1)
rng=np.random.default_rng(SEED);yp=yt.copy();buckets={}
for i,r in enumerate(train):buckets.setdefault(tuple(r['targets']),[]).append(i)
for idx in buckets.values():yp[idx]=rng.permutation(yt[idx])

def feat(rr,site,u):
 with safe_open(str(ST),framework='pt',device='cpu') as f:
  return np.stack([f.get_tensor(r['split']+'_'+hashlib.sha256(r['id'].encode()).hexdigest()+f'_u{u}_'+site).float().numpy().ravel().astype(np.float64) for r in rr])
def loss(w,X,mask,y,lam):
 w=w.reshape(X.shape[1],C);logits=X@w;logits[~mask]=-np.inf;ls=logsumexp(logits,axis=1);v=np.mean(ls-logits[np.arange(len(y)),y])+0.5*lam*np.sum(w**2);p=np.exp(logits-ls[:,None]);p[np.arange(len(y)),y]-=1;g=X.T@p/len(y);g+=lam*w;return v,g.ravel()
def calc(X,mask,y,w,rr):
 logits=X@w;logits[~mask]=-np.inf;ls=logsumexp(logits,axis=1);valid=y>=0;prob=np.exp(logits-ls[:,None]);item=[];correct=[];ll=[];confidence=[];goldprob=[];brier=[]
 for i,r in enumerate(rr):
  if not valid[i]:item.append({'id':r['id'],'status':'UNSUPPORTED_CANDIDATE_OOV','targets':r['targets']});continue
  values=[float(logits[i,ci[t]]) for t in r['targets']]; pp=[float(prob[i,ci[t]]) for t in r['targets']];best=max(values);idx=next(j for j,z in enumerate(values) if z==best);tied=[j for j,z in enumerate(values) if z==best]; correct.append(idx==r['gold']);ll.append(float(ls[i]-logits[i,y[i]]));confidence.append(max(pp));goldprob.append(float(prob[i,y[i]]));brier.append(float(sum((z-(j==r['gold']))**2 for j,z in enumerate(pp))));item.append({'candidate_gap':[values[r['gold']]-values[j] for j in range(len(values)) if j!=r['gold']],'dynamic_gap':values[r['gold']]-max(values[j] for j in range(len(values)) if j!=r['gold']),'id':r['id'],'status':'OK','targets':r['targets'],'gold_index':r['gold'],'candidate_logits':values,'candidate_probabilities':pp,'prediction_index':idx,'top_ties':tied,'group':r['group'],'natural_no_top_tie':r['natural_no_top_tie']})
 return {'N':len(y),'supported':len(correct),'oov_ids':[z['id'] for z in item if z['status']!='OK'],'accuracy':float(np.mean(correct)) if correct else None,'logloss':float(np.mean(ll)) if ll else None,'mean_confidence':float(np.mean(confidence)) if confidence else None,'mean_gold_probability':float(np.mean(goldprob)) if goldprob else None,'brier':float(np.mean(brier)) if brier else None,'ece_10_fixed_bins':float(sum(sum(1 for z in confidence if lo<=z<(lo+.1) or (lo==.9 and z==1.0))/len(confidence)*abs(np.mean([correct[i] for i,z in enumerate(confidence) if lo<=z<(lo+.1) or (lo==.9 and z==1.0)])-np.mean([z for z in confidence if lo<=z<(lo+.1) or (lo==.9 and z==1.0)])) for lo in [j/10 for j in range(10)] if any(lo<=z<(lo+.1) or (lo==.9 and z==1.0) for z in confidence))) if confidence else None,'questions':item}

output=E/'probe_fit_attempt3';output.mkdir(exist_ok=False);metadata={'status':'E2_TRAIN_VAL_PROBE_SELECTION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'no test predictions or labels were evaluated; previous ARC outcomes exposed; learned decodability is not natural use','train_N':len(train),'validation_N':len(val),'classes_target_ids':classes,'penalties':LAMBDA,'solver':'scipy L-BFGS-B','ftol':1e-9,'gtol':1e-6,'maxiter':4000,'seed':SEED,'bias_penalty':'same lambda as weights','shared_transform':'pooled K1 and K2 non-demo train only, separate feature type/site','source_audit_sha256':sha(AUDIT),'state_sha256':sha(ST),'source_rows_sha256':sha(ROWS),'order_sha256':sha(ORDER),'script_sha256':sha(Path(__file__)),'models':{}}
def prior_baseline(rr,mode):
 global_counts=np.ones(C);counts_by_n={}
 for r,y in zip(train,yt):
  global_counts[y]+=1;counts_by_n.setdefault(len(r['targets']),np.ones(C))[y]+=1
 out=[]
 for r in rr:
  if not all(t in ci for t in r['targets']):out.append({'id':r['id'],'status':'UNSUPPORTED_CANDIDATE_OOV'});continue
  counts=counts_by_n.get(len(r['targets'])) if mode=='candidate_count' else global_counts
  if counts is None:out.append({'id':r['id'],'status':'UNSUPPORTED_CANDIDATE_COUNT'});continue
  vals=np.ones(len(r['targets'])) if mode=='uniform' else np.array([counts[ci[t]] for t in r['targets']]);prob=vals/vals.sum();idx=int(np.argmax(prob));out.append({'id':r['id'],'status':'OK','candidate_probabilities':prob.tolist(),'prediction_index':idx,'top_ties':[j for j,v in enumerate(prob) if v==max(prob)],'gold_index':r['gold'],'group':r['group'],'natural_no_top_tie':r['natural_no_top_tie']})
 return out
metadata['baselines']={mode:{'train':prior_baseline(train,mode),'validation':prior_baseline(val,mode)} for mode in ('uniform','global','candidate_count')}
arrays={}
for site in ('l18_input','post_norm'):
 t=[feat(train,site,u) for u in (0,1)];v=[feat(val,site,u) for u in (0,1)]
 for typ in ('raw','unit_l2'):
  if typ=='unit_l2':
   def unit(a):
    n=np.linalg.norm(a,axis=1,keepdims=True);return np.divide(a,n,out=np.zeros_like(a),where=n!=0)
   X=[unit(a) for a in t];V=[unit(a) for a in v]
  else:X=[a.copy() for a in t];V=[a.copy() for a in v]
  pooled=np.concatenate(X);mu=pooled.mean(axis=0);std=pooled.std(axis=0);std[std==0]=1;X=[np.column_stack(((a-mu)/std,np.ones(len(a)))) for a in X];V=[np.column_stack(((a-mu)/std,np.ones(len(a)))) for a in V]
  for u in (0,1):
   for control in ('real','permuted'):
    y=yt if control=='real' else yp; fits=[]
    for lam in LAMBDA:
     z=minimize(loss,np.zeros(X[u].shape[1]*C),args=(X[u],mt,y,lam),jac=True,method='L-BFGS-B',options={'maxiter':4000,'ftol':1e-9,'gtol':1e-6});assert z.success and np.isfinite(z.fun) and np.isfinite(z.x).all(), f'optimizer failed {site} {typ} u={u} {control} lambda={lam}: {z.message}';w=z.x.reshape(X[u].shape[1],C);vr=calc(V[u],mv,yv,w,val);fits.append((vr['logloss'],lam,w,{'lambda':lam,'train_loss':float(z.fun),'validation_same_loop_logloss':vr['logloss'],'optimizer_success':bool(z.success),'iterations':int(z.nit),'message':str(z.message)}))
    assert val_supported.any(); best=min(fits,key=lambda q:(q[0],-q[1]));name=f'{site}_{typ}_trainK{u+1}_{control}';arrays[name+'_w']=best[2];arrays[name+'_mu']=mu;arrays[name+'_std']=std; metadata['models'][name]={'selected_lambda':best[1],'fits':[p[3] for p in fits],'train_K1':calc(X[0],mt,yt,best[2],train),'train_K2':calc(X[1],mt,yt,best[2],train),'validation_K1':calc(V[0],mv,yv,best[2],val),'validation_K2':calc(V[1],mv,yv,best[2],val)};print('FIT',name,'selected',best[1],flush=True)
npz=output/'selected_models.npz';np.savez_compressed(npz,**arrays);metadata['models_sha256']=sha(npz);assert sha(ST)==metadata['state_sha256'] and sha(ROWS)==metadata['source_rows_sha256'] and sha(ORDER)==metadata['order_sha256'] and sha(AUDIT)==metadata['source_audit_sha256'];p=output/'train_val_selection.json'
with p.open('x') as f:json.dump(metadata,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
print('PROBE_SELECTION_FROZEN',sha(p),flush=True)
