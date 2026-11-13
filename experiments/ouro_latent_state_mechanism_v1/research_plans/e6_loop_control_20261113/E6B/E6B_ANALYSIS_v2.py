"""E6B strict Train/Val development freeze, then separate frozen Test evaluation."""
import argparse,json,time,importlib.util,hashlib
from pathlib import Path
import numpy as np
from E6B_ANALYSIS import fit,apply,macro,diagnostic,controller_metrics,sha,write
from E6B_TRAINING_FREE_CONTROLLER import predict,threshold_from_unlabeled_train

def load(captures,labels):
 data=[];sources=[];cores=[]
 for directory in captures:
  directory=Path(directory);identity=json.loads((directory/'IDENTITY.json').read_text(encoding='utf-8-sig'));core={k:v for k,v in identity.items() if k!='inputs_sha256'};cores.append(core)
  for f in sorted(directory.glob('*.json')):
   r=json.loads(f.read_text(encoding='utf-8-sig'))
   if 'native' not in r:continue
   assert r['identity']==identity;l=labels[(r['split'],r['id'])]
   z={'id':r['id'],'split':r['split'],'gold':l['gold'],'is_demonstration':l['is_demonstration'],'scores':[n['scores'] for n in r['native']],'pred':[n['prediction'] for n in r['native']],'ties':[n['top_tie'] for n in r['native']]}
   assert len(z['scores'])==4 and all(int(np.argmax(s))==p for s,p in zip(z['scores'],z['pred']))
   for family in ['latent','unrelated']:z[family]=[next(x['values'] for x in r['features'] if x['family']==family and x['depth']==d) for d in range(1,5)]
   z['replica_baselines']=r.get('replica_baselines',{})
   data.append(z);sources.append({'path':str(f),'sha256':sha(f)})
 assert all(core==cores[0] for core in cores) and len({(r['split'],r['id']) for r in data})==len(data)
 return {s:[r for r in data if r['split']==s and not r['is_demonstration']] for s in ['train','validation','test']},sources,cores[0]

def decisions(rows,spec,thresholds):return [predict(r['scores'],r['latent'],thresholds[str(spec['q'])],spec['strategy'],spec['use_latent'],spec['p']) for r in rows]
def evaluate(splits,specs,thresholds,a,asp,amod):
 results=[];individual=[]
 for split,rows in splits.items():
  if not rows:continue
  bases={'K4':[r['pred'][3] for r in rows]}
  for name in ['primary_ZL','primary_Cal']:
   methodname=asp[name];bases[name]=[r['replica_baselines'][methodname]['prediction'] if methodname in r['replica_baselines'] else amod.infer(r['scores'],asp['methods'][methodname])['prediction'] for r in rows]
  for name in ['loopcd_adaptive_cap1','loopcd_fixed_0.5']:
   if all(name in r['replica_baselines'] for r in rows):bases[name]=[r['replica_baselines'][name]['prediction'] for r in rows]
  for index,spec in enumerate(specs):
   dec=decisions(rows,spec,thresholds);m=controller_metrics(rows,dec,bases['K4']);m.update({'split':split,'method':spec,'method_index':index,'relative_frozen_E6A':{name:controller_metrics(rows,dec,bases[name]) for name in bases if name!='K4'}})
   subset=[i for i,r in enumerate(rows) if not any(r['ties'])]
   if subset:m['native_all_depth_no_tie']=controller_metrics([rows[i] for i in subset],[dec[i] for i in subset],[bases['K4'][i] for i in subset])
   results.append(m)
   for r,d in zip(rows,dec):individual.append({'split':split,'id':r['id'],'gold':r['gold'],'method_index':index,'permission':spec['track'],**d})
 return results,individual

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['develop','evaluate'],required=True);p.add_argument('--capture',nargs='+',required=True);p.add_argument('--labels',required=True);p.add_argument('--test-inputs',required=True);p.add_argument('--out',required=True);p.add_argument('--e6a-methods',required=True);p.add_argument('--e6a-spec',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=True,parents=True);start=time.time();labels={(r['split'],r['id']):r for r in map(json.loads,Path(a.labels).read_text(encoding='utf-8-sig').splitlines())};splits,sources,core=load(a.capture,labels)
 im=importlib.util.spec_from_file_location('e6a_frozen',a.e6a_methods);amod=importlib.util.module_from_spec(im);im.loader.exec_module(amod);asp=json.loads(Path(a.e6a_spec).read_text(encoding='utf-8-sig'))
 hashes={'analysis_source_sha256':sha(__file__),'metric_helper_sha256':sha(Path(__file__).parent/'E6B_ANALYSIS.py'),'controller_source_sha256':sha(Path(__file__).parent/'E6B_TRAINING_FREE_CONTROLLER.py'),'e6a_methods_sha256':sha(a.e6a_methods),'e6a_spec_sha256':sha(a.e6a_spec),'test_inputs_sha256':sha(a.test_inputs),'labels_sha256':sha(a.labels)}
 freeze_path=out/'E6B_FROZEN_SPEC.json'
 if a.phase=='develop':
  assert [len(splits[s]) for s in ['train','validation','test']]==[1094,299,0] and not freeze_path.exists()
  diagnostic_specs=[];model_arrays={};i=0
  for setting in ['retrospective','prospective']:
   for family in ['output','latent','combined','unrelated','shuffled']:
    tr,va=splits['train'],splits['validation'];xt,yt,_=diagnostic(tr,setting,family,np.random.default_rng(31).permutation(len(tr)));xv,yv,_=diagnostic(va,setting,family,np.random.default_rng(32).permutation(len(va)));candidate=[]
    for alpha in [.1,1.,10.]:
     model=fit(xt,yt,alpha);candidate.append((macro(yv,apply(model,xv)),-alpha,model))
    score,neg_alpha,model=max(candidate,key=lambda z:(z[0],z[1]));diagnostic_specs.append({'setting':setting,'family':family,'track':'SUPERVISED_DIAGNOSTIC','alpha':-neg_alpha,'validation_macro_f1':score})
    for key,value in zip(['mean','scale','coef'],model):model_arrays[str(i)+'_'+key]=value
    i+=1
  thresholds={str(q):threshold_from_unlabeled_train(splits['train'],q) for q in [.25,.5,.75]};specs=[{'strategy':s,'use_latent':b,'q':.5,'p':.9,'track':'ZL'} for s in ['rollback','agreement','stability','early_exit'] for b in [False,True]]
  grid=[{'strategy':s,'use_latent':b,'q':q,'p':p,'track':'Cal'} for s in ['rollback','early_exit'] for b in [False,True] for q in ([.25,.5,.75] if b else [.5]) for p in ([.9] if s=='rollback' else [.8,.9,.95])];chosen=[]
  for latent in [False,True]:
   cand=[s for s in grid if s['use_latent']==latent];index=max(range(len(cand)),key=lambda i:(sum(d['prediction']==r['gold'] for d,r in zip(decisions(splits['validation'],cand[i],thresholds),splits['validation'])),-i));chosen.append(cand[index])
  np.savez_compressed(out/'SUPERVISED_DIAGNOSTIC_MODELS.npz',**model_arrays)
  frozen={'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'phase':'DEV_FROZEN_BEFORE_NEW_TEST_CAPTURE','hashes':hashes,'capture_identity_core':core,'development_sources':sources,'diagnostic_model_arrays_sha256':sha(out/'SUPERVISED_DIAGNOSTIC_MODELS.npz'),'diagnostics':diagnostic_specs,'thresholds_unlabeled_train':thresholds,'zero_label_specs':specs,'calibration_grid':grid,'selected_cal_specs':chosen,'primary_ZL':'rollback_use_latent_true','metric':'Test accuracy paired versus same-device K4 and two E6A frozen primary methods','test_exposure':'ARC historically exposed; no new Test capture or method outputs consumed in development.'};write(freeze_path,frozen)
  result,individual=evaluate({s:splits[s] for s in ['train','validation']},specs+chosen,thresholds,a,asp,amod);write(out/'E6B_DEVELOPMENT_SUMMARY.json',result);(out/'E6B_DEVELOPMENT_DECISIONS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in individual),encoding='utf8');print('E6B_DEV_FROZEN',sha(freeze_path),flush=True)
 else:
  frozen=json.loads(freeze_path.read_text());assert hashes==frozen['hashes'] and core==frozen['capture_identity_core'];assert not (out/'E6B_TEST_RECEIPT.json').exists();assert len(splits['test'])==1172 and not splits['train'] and not splits['validation'];expected={(r['split'],r['id']) for r in map(json.loads,Path(a.test_inputs).read_text().splitlines())};assert {(r['split'],r['id']) for r in splits['test']}==expected;assert sha(out/'SUPERVISED_DIAGNOSTIC_MODELS.npz')==frozen['diagnostic_model_arrays_sha256']
  result,individual=evaluate({'test':splits['test']},frozen['zero_label_specs']+frozen['selected_cal_specs'],frozen['thresholds_unlabeled_train'],a,asp,amod);coeff=np.load(out/'SUPERVISED_DIAGNOSTIC_MODELS.npz');diag=[]
  for i,s in enumerate(frozen['diagnostics']):
   model=tuple(coeff[str(i)+'_'+key] for key in ['mean','scale','coef']);rs=splits['test'];x,y,_=diagnostic(rs,s['setting'],s['family'],np.random.default_rng(33).permutation(len(rs)));pr=apply(model,x);diag.append({**s,'test_accuracy':float((pr==y).mean()),'test_macro_f1':macro(y,pr)})
  write(out/'E6B_DIAGNOSTICS.json',diag);write(out/'E6B_TEST_SUMMARY.json',result);(out/'E6B_CONTROLLER_RESULTS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in individual),encoding='utf8');write(out/'E6B_TEST_RECEIPT.json',{'status':'COMPLETE','n':1172,'frozen_spec_sha256':sha(freeze_path),'sources':sources,'seconds':time.time()-start,'new_fits':0});print('E6B_TEST_COMPLETE',len(individual),flush=True)
if __name__=='__main__':main()
