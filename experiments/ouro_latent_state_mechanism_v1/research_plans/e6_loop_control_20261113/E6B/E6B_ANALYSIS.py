"""Train/Validation freeze then one Test evaluation; no collector/model imports."""
import argparse,json,hashlib,time,importlib.util
from pathlib import Path
import numpy as np
from E6B_TRAINING_FREE_CONTROLLER import predict,output_features,threshold_from_unlabeled_train

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def macro(y,p):
 vals=[]
 for c in range(4):
  tp=np.sum((y==c)&(p==c));den=np.sum(y==c)+np.sum(p==c);vals.append(2*tp/den if den else 0.)
 return float(np.mean(vals))
def utility(a,b,g):return 3 if a==g and b==g else 2 if a==g else 1 if b==g else 0

def fit(x,y,alpha):
 mu=x.mean(0);sd=x.std(0);sd[sd<1e-8]=1.;z=np.c_[(x-mu)/sd,np.ones(len(x))];counts=np.bincount(y,minlength=4);w=1/np.maximum(counts[y],1);w=w/w.mean();pen=np.eye(z.shape[1])*alpha;pen[-1,-1]=0.;coef=np.linalg.solve(z.T@(w[:,None]*z)+pen,z.T@(w[:,None]*np.eye(4)[y]));return mu,sd,coef
def apply(model,x):mu,sd,coef=model;return np.argmax(np.c_[(x-mu)/sd,np.ones(len(x))]@coef,axis=1)
def diagnostic(rows,setting,family,permutation=None):
 xx=[];yy=[];ids=[]
 for i,r in enumerate(rows):
  for d in ([2,3,4] if setting=='retrospective' else [2,3]):
   before=d-1 if setting=='retrospective' else d;after=d if setting=='retrospective' else d+1;k=d-1
   latent=np.array(r['latent'][k]);control=np.array(r['unrelated'][k]);out=output_features(r['scores'],d)
   if family=='shuffled':latent=np.array(rows[permutation[i]]['latent'][k])
   x={'output':out,'latent':latent,'combined':np.r_[out,latent],'unrelated':np.r_[out,control],'shuffled':np.r_[out,latent]}[family]
   xx.append(x);yy.append(utility(r['pred'][before-1],r['pred'][after-1],r['gold']));ids.append(r['id'])
 return np.array(xx),np.array(yy),ids

def controller_metrics(rows,decisions,base):
 y=np.array([r['gold'] for r in rows]);p=np.array([d['prediction'] for d in decisions]);b=np.array(base);delta=(p==y).astype(float)-(b==y);rng=np.random.default_rng(20261113);means=[]
 for _ in range(100):means.extend(delta[rng.integers(0,len(rows),(100,len(rows)))].mean(1))
 return {'n':len(rows),'correct':int((p==y).sum()),'accuracy':float((p==y).mean()),'rescue':int(((p==y)&(b!=y)).sum()),'damage':int(((p!=y)&(b==y)).sum()),'net_gain':int(delta.sum()),'pp_gain':float(delta.mean()*100),'paired_question_bootstrap95_pp':(np.quantile(means,[.025,.975])*100).tolist(),'mean_executed_loops':float(np.mean([d['executed_loops'] for d in decisions])),'online_latency':'NOT_RUN'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--labels',required=True);p.add_argument('--out',required=True);p.add_argument('--e6a-methods',required=True);p.add_argument('--e6a-spec',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);assert not (out/'E6B_FROZEN_SPEC.json').exists(),'Frozen Test namespace already exists; do not refit or overwrite';start=time.time()
 labels={(r['split'],r['id']):r for r in map(json.loads,Path(a.labels).read_text(encoding='utf-8-sig').splitlines())};data=[];source=[]
 for f in sorted(Path(a.capture).glob('*.json')):
  r=json.loads(f.read_text(encoding='utf-8-sig'))
  if 'native' not in r:continue
  l=labels[(r['split'],r['id'])];assert len(r['native'])==4
  source.append({'path':str(f),'sha256':sha(f)});z={'id':r['id'],'split':r['split'],'gold':l['gold'],'is_demonstration':l['is_demonstration'],'scores':[n['scores'] for n in r['native']],'pred':[n['prediction'] for n in r['native']],'ties':[n['top_tie'] for n in r['native']]}
  for fam in ['latent','unrelated']:z[fam]=[next(x['values'] for x in r['features'] if x['family']==fam and x['depth']==d) for d in range(1,5)]
  assert all(int(np.argmax(n['scores']))==n['prediction'] for n in r['native'])
  assert r['identity']==json.loads((Path(a.capture)/'IDENTITY.json').read_text(encoding='utf-8-sig'))
  data.append(z)
 split={s:[r for r in data if r['split']==s and not r['is_demonstration']] for s in ['train','validation','test']};assert [len(split[s]) for s in split]==[1094,299,1172]
 # No Test outcome is used above, and none enters the following fit/selection.
 diagnostics=[];models=[]
 for setting in ['retrospective','prospective']:
  for family in ['output','latent','combined','unrelated','shuffled']:
   tr=split['train'];va=split['validation'];xt,yt,_=diagnostic(tr,setting,family,np.random.default_rng(31).permutation(len(tr)));xv,yv,_=diagnostic(va,setting,family,np.random.default_rng(32).permutation(len(va)));candidates=[]
   for alpha in [.1,1.,10.]:
    model=fit(xt,yt,alpha);pr=apply(model,xv);candidates.append((macro(yv,pr),-alpha,model))
   score,neg_alpha,model=max(candidates,key=lambda t:(t[0],t[1]));models.append((setting,family,model));diagnostics.append({'setting':setting,'family':family,'permission':'SUPERVISED_DIAGNOSTIC','alpha':-neg_alpha,'validation_macro_f1':score})
 qs=[.25,.5,.75];thresholds={str(q):threshold_from_unlabeled_train(split['train'],q) for q in qs};specs=[{'strategy':s,'use_latent':b,'q':.5,'p':.9,'track':'ZL'} for s in ['rollback','agreement','stability','early_exit'] for b in [False,True]]
 cal=[{'strategy':s,'use_latent':b,'q':q,'p':p,'track':'Cal'} for s in ['rollback','early_exit'] for b in [False,True] for q in (qs if b else [.5]) for p in ([.9] if s=='rollback' else [.8,.9,.95])]
 def infer(rows,spec):return [predict(r['scores'],r['latent'],thresholds[str(spec['q'])],spec['strategy'],spec['use_latent'],spec['p']) for r in rows]
 selected=[]
 for latent in [False,True]:
  cand=[s for s in cal if s['use_latent']==latent];best=max(enumerate(cand),key=lambda z:(sum(d['prediction']==r['gold'] for d,r in zip(infer(split['validation'],z[1]),split['validation'])),-z[0]))[1];selected.append(best)
 model_arrays={}
 for i,(_,_,m) in enumerate(models):
  for key,value in zip(['mean','scale','coef'],m):model_arrays[str(i)+'_'+key]=value
 np.savez_compressed(out/'SUPERVISED_DIAGNOSTIC_MODELS.npz',**model_arrays)
 freeze={'diagnostic_model_arrays_sha256':sha(out/'SUPERVISED_DIAGNOSTIC_MODELS.npz'),'utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'source_sha256':sha(__file__),'controller_source_sha256':sha(Path(__file__).parent/'E6B_TRAINING_FREE_CONTROLLER.py'),'capture_sources':source,'thresholds_unlabeled_train':thresholds,'zero_label_specs':specs,'calibration_grid':cal,'selected_cal_specs':selected,'diagnostics':diagnostics,'primary_zl':'rollback_use_latent_true','metric':'accuracy relative same-device K4 and frozen E6A','test_exposure':'Historically exposed ARC; no new Test controller outputs used in this freeze.'};write(out/'E6B_FROZEN_SPEC.json',freeze)
 am=importlib.util.spec_from_file_location('e6a_frozen',a.e6a_methods);amod=importlib.util.module_from_spec(am);am.loader.exec_module(amod);asp=json.loads(Path(a.e6a_spec).read_text());freeze['e6a_source_sha256']=sha(a.e6a_methods);freeze['e6a_frozen_spec_sha256']=sha(a.e6a_spec);write(out/'E6B_FROZEN_SPEC.json',freeze)
 results=[];individual=[]
 for s,rows in split.items():
  base=[r['pred'][3] for r in rows]
  for index,spec in enumerate(specs+selected):
   decisions=infer(rows,spec);m=controller_metrics(rows,decisions,base);m.update({'split':s,'method':spec,'index':index});m['relative_frozen_E6A']={name:controller_metrics(rows,decisions,[amod.infer(r['scores'],asp['methods'][asp[name]])['prediction'] for r in rows]) for name in ['primary_ZL','primary_Cal']};results.append(m)
   for r,d in zip(rows,decisions):individual.append({'split':s,'id':r['id'],'gold':r['gold'],'method_index':index,'permission':spec['track'],**d})
 for item,(setting,family,model) in zip(diagnostics,models):
  te=split['test'];xt,yt,_=diagnostic(te,setting,family,np.random.default_rng(33).permutation(len(te)));pr=apply(model,xt);item['test_accuracy']=float((pr==yt).mean());item['test_macro_f1']=macro(yt,pr)
 write(out/'E6B_DIAGNOSTICS.json',diagnostics);write(out/'E6B_SUMMARY.json',results);(out/'E6B_CONTROLLER_RESULTS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in individual),encoding='utf8');write(out/'E6B_CPU_RUNTIME.json',{'seconds':time.time()-start,'scope':'CPU feature loading, diagnostic training/validation, frozen Test diagnostics/controllers; no GPU forwards','all_results_after_freeze_sha256':sha(out/'E6B_FROZEN_SPEC.json')});print('E6B_CPU_COMPLETE',len(individual))
if __name__=='__main__':main()
