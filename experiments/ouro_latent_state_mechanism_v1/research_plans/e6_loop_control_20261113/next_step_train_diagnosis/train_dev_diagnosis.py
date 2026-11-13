"""Train-only, question-group-disjoint failure diagnosis. No Test/Val inputs or GPU."""
import json,hashlib,time,sys,collections,datetime
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/ouro-e6a');O=R/'next_step_train_diagnosis';sys.path.insert(0,str(R/'E6B'))
from E6B_ANALYSIS import fit,macro,utility
from E6B_TRAINING_FREE_CONTROLLER import output_features
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(n,x):(O/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def fold(key,domain,n):return int(hashlib.sha256((domain+key).encode()).hexdigest(),16)%n
def array(rows,setting,family,seed):
 xs=[];ys=[];ids=[];depths=[];perm=np.random.default_rng(seed).permutation(len(rows))
 for i,r in enumerate(rows):
  for d in ([2,3,4] if setting=='retrospective' else [2,3]):
   before=d-1 if setting=='retrospective' else d;after=d if setting=='retrospective' else d+1;o=output_features(r['scores'],d)
   b=np.array(rows[perm[i]]['latent'][d-1] if family=='shuffled' else r['unrelated'][d-1] if family=='unrelated' else r['latent'][d-1])
   xs.append(o if family=='output' else np.r_[o,b]);ys.append(utility(r['pred'][before-1],r['pred'][after-1],r['gold']));ids.append(r['id']);depths.append(d)
 return np.array(xs),np.array(ys),ids,depths
def probs(m,x):
 mu,sd,c=m;z=np.c_[(x-mu)/sd,np.ones(len(x))]@c;z-=z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)
def losses(p,y):return ((p-np.eye(4)[y])**2).sum(1),-np.log(np.maximum(p[np.arange(len(y)),y],1e-30))
def metrics(p,y):
 b,l=losses(p,y);pr=p.argmax(1);conf=p.max(1);ece=0.
 for a in np.arange(0,1,.1):
  m=(conf>=a)&(conf<a+.1 if a<.9 else conf<=1)
  if m.any():ece+=m.mean()*abs(conf[m].mean()-(pr[m]==y[m]).mean())
 return {'n_transitions':len(y),'brier':float(b.mean()),'log_loss':float(l.mean()),'utility_accuracy':float((pr==y).mean()),'macro_f1':macro(y,pr),'ECE10':float(ece)}
def bootstrap(d):
 d=np.array(d);rng=np.random.default_rng(20261113);values=[]
 for _ in range(100):values.extend(d[rng.integers(0,len(d),(100,len(d)))].mean(1))
 return np.quantile(values,[.025,.975]).tolist()
def main():
 t=time.perf_counter();O.mkdir(exist_ok=True);fixture=O/'TRAIN_ONLY_INPUTS.jsonl';assert fixture.exists(),'Use isolated exporter before fitting; no all-split label files in this process'
 rows=[json.loads(l) for l in fixture.read_text().splitlines()];assert len(rows)==1094 and all(r['split']=='train' and not r['is_demonstration'] for r in rows)
 counts=collections.Counter(r['outer_fold'] for r in rows);freeze=O/'DIAGNOSIS_FREEZE.json';assert not freeze.exists(),'Do not overwrite new-operator freeze'
 put('DIAGNOSIS_FREEZE.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(__file__),'train_fixture_sha256':sha(fixture),'director_sha256':sha(R/'ASTRA_NEXT_STEP_DIRECTIVE.md'),'head':'Original class-balanced ridge, same standardization; softmax logits temperature1','alphas':[.1,1.,10.],'primary':'4-class Brier (sum squared errors; lower better)','secondary':['logloss','utility macroF1','ECE10','closed-loop diagnostic action utility'],'families':['output','combined','unrelated','shuffled'],'outer':'5 hash folds, exact/known Train near duplicates same group','inner':'3 disjoint hash folds, only outer-training questions','family_dim':{'output':10,'combined':26,'unrelated':26,'shuffled':26},'complexity_note':'Same estimator and regularization, matched16 extra dimensions for latent/unrelated/shuffle; output-only has10 inputs, not falsely equal parameter count.','outcomes_seen_before_freeze':False,'data_policy':'No Test or Validation examples loaded by this process; no method promotion from Test','fold_counts':dict(counts),'new_GPU_forwards':0})
 put('FOLD_MANIFEST.json',[{k:r[k] for k in ['id','group','outer_fold','inner_fold','input_ids_sha256']} for r in rows])
 pred=[];summary=[];alphas=[];byid={r['id']:r for r in rows}
 for setting in ['retrospective','prospective']:
  for family in ['output','combined','unrelated','shuffled']:
   allp=[];ally=[];alli=[];alld=[];allf=[]
   for outer in range(5):
    tr=[r for r in rows if r['outer_fold']!=outer];va=[r for r in rows if r['outer_fold']==outer];scores=[]
    for alpha in [.1,1.,10.]:
     total=0.;n=0
     for inner in range(3):
      ti=[r for r in tr if r['inner_fold']!=inner];vi=[r for r in tr if r['inner_fold']==inner];xt,yt,_,_=array(ti,setting,family,1000+outer*10+inner);xv,yv,_,_=array(vi,setting,family,2000+outer*10+inner);m=fit(xt,yt,alpha);loss,_=losses(probs(m,xv),yv);total+=loss.sum();n+=len(loss)
     scores.append((total/n,alpha))
    alpha=min(scores)[1];xt,yt,_,_=array(tr,setting,family,3000+outer);xv,yv,ids,depths=array(va,setting,family,4000+outer);m=fit(xt,yt,alpha);pr=probs(m,xv);allp.extend(pr);ally.extend(yv);alli.extend(ids);alld.extend(depths);allf.extend([outer]*len(ids));alphas.append({'setting':setting,'family':family,'outer_fold':outer,'selected_alpha':alpha,'inner_scores':scores})
   pp=np.array(allp);yy=np.array(ally);summary.append({'setting':setting,'family':family,**metrics(pp,yy)})
   b,l=losses(pp,yy)
   for i,key in enumerate(alli):pred.append({'id':key,'setting':setting,'family':family,'depth':alld[i],'outer_fold':allf[i],'utility_class':int(yy[i]),'probabilities':pp[i].tolist(),'utility_prediction':int(pp[i].argmax()),'brier':float(b[i]),'log_loss':float(l[i])})
   print('OOF_DONE',setting,family,summary[-1],flush=True)
 (O/'OOF_PREDICTIONS.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in pred));put('OOF_SUMMARY.json',summary);put('NESTED_ALPHA_SELECTION.json',alphas)
 comparisons=[]
 for setting in ['retrospective','prospective']:
  z={fam:{} for fam in ['output','combined','unrelated','shuffled']}
  for p in pred:
   if p['setting']==setting:z[p['family']].setdefault(p['id'],[]).append(p['brier'])
  for family in ['combined','unrelated','shuffled']:
   ids=sorted(z['output']);d=[np.mean(z['output'][k])-np.mean(z[family][k]) for k in ids];comparisons.append({'setting':setting,'family':family,'n_questions':len(ids),'Brier_improvement_vs_output':float(np.mean(d)),'paired_question_bootstrap95':bootstrap(d)})
 put('CONDITIONAL_INFORMATION_COMPARISONS.json',comparisons)
 # Closed-loop diagnostic actions are fixed mappings, supervised and exploratory, not a ZL inference method.
 actions=[]
 for setting in ['retrospective','prospective']:
  for family in ['output','combined','unrelated','shuffled']:
   cache={(p['id'],p['depth']):p['utility_prediction'] for p in pred if p['setting']==setting and p['family']==family};correct=[];base=[];depth=[]
   for r in rows:
    d=3 if setting=='retrospective' and cache[(r['id'],4)]==2 else 4
    if setting=='prospective':
     for k in [2,3]:
      if cache[(r['id'],k)] in [2,3]:d=k;break
    correct.append(r['pred'][d-1]==r['gold']);base.append(r['pred'][3]==r['gold']);depth.append(d if setting=='prospective' else 4)
   delta=np.array(correct,dtype=int)-np.array(base,dtype=int);actions.append({'setting':setting,'family':family,'n':len(rows),'correct':sum(correct),'native_K4_correct':sum(base),'net_correct':int(delta.sum()),'rescue':int((delta==1).sum()),'damage':int((delta==-1).sum()),'mean_replay_loops':float(np.mean(depth)),'diagnostic_permission':'SUPERVISED OOF, not ZL/Cal and not a new selected controller','actual_online_latency':'NOT RUN'})
 put('FIXED_ACTION_DIAGNOSTICS.json',actions)
 c=np.array([[p==r['gold'] for p in r['pred']] for r in rows]);h0={'n':len(rows),'native_correct':c.sum(0).tolist(),'oracle_any_depth_correct':int(c.any(1).sum()),'K4_wrong_previous_correct':int((~c[:,3]&c[:,:3].any(1)).sum()),'K4_correct_K2_wrong':int((c[:,3]&~c[:,1]).sum()),'all_native_wrong':int((~c.any(1)).sum()),'oracle_is_not_a_method':True};put('TRAIN_ACTION_HEADROOM.json',h0)
 md='# Train-only failure diagnosis\n\nAll analysis and code: current Sol. Astra supplies only the separately frozen directive. No GPU forwards; no Validation/Test examples loaded by the fit process. Existing exposed Train is not independent benchmark confirmation. Exact inputs share a fold; the known Train near-duplicate pair is exact and grouped. All depths of each question stay together.\n\n## Cache limitation\n\nNatural attention increments are stored per layer, but residual x is averaged across seven layers. The per-layer final-normalization lens in the directive is therefore NOT COMPUTABLE from the current cache. Mean-x is not substituted. This analysis does not evaluate H2 or imply its failure.\n\n## Train action upper bounds (offline gold only)\n\n'+json.dumps(h0,indent=2)+'\n\n## Conditional information\n\n'+json.dumps(comparisons,indent=2)+'\n\nA positive Brier improvement means adding the feature block reduces proper predictive loss. Intervals resample whole questions, with every depth together. Inner/outer fits never share a question group; alpha selection uses inner Brier only. Same ridge head, same3 alpha choices, fixed temperature1; input dimensions are10 versus26, with same-dimension unrelated/shuffle controls. Probability maps are uncalibrated ridge-softmax; logloss and ECE are retained. Correlated CV training and historical mechanism selection limit generalization claims. No controller is tuned from these OOF outcomes.\n\nFixed supervised action mappings and all OOF predictions are included, but they do not establish a deployable early-exit algorithm or latency benefit. The H2 family remains a separately preregistered hypothesis requiring true per-layer x/a and native endpoint validation; E6C remains NO-GO.\n'
 (O/'TRAIN_DEV_FAILURE_DIAGNOSIS.md').write_text(md);put('COMPLETE.json',{'status':'COMPLETE','cpu_wall_seconds':time.perf_counter()-t,'new_GPU_forwards':0,'question_count':len(rows),'fit_inputs_sha256':sha(fixture),'source_sha256':sha(__file__),'freeze_sha256':sha(freeze),'all_predictions':len(pred)});print('TRAIN_DIAGNOSIS_COMPLETE',time.perf_counter()-t,flush=True)
if __name__=='__main__':main()
