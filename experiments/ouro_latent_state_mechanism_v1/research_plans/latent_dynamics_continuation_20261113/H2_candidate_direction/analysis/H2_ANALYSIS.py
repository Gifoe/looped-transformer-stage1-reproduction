"""H2 supervised diagnostic, original estimator/seeds; Train freeze then one exposed-Val review."""
import argparse,json,hashlib,datetime,time,sys,importlib.util,collections
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def jsonl(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
def putrows(p,rows):Path(p).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
def load_base(root):
 f=root/'train_dev_diagnosis.py';im=importlib.util.spec_from_file_location('original_train_diagnosis',f);m=importlib.util.module_from_spec(im);im.loader.exec_module(m);return m
def hashes(a):
 r=Path(a.e6);e=Path(a.root)
 return {'analysis_source':sha(__file__),'base_diagnosis_source':sha(r/'train_dev_diagnosis.py'),'estimator_source':sha(r/'E6B/E6B_ANALYSIS.py'),'output_feature_source':sha(r/'E6B/E6B_TRAINING_FREE_CONTROLLER.py'),'preregistration':sha(e/'H2_PRE_REGISTRATION.json'),'train_fixture':sha(r/'next_step_train_diagnosis/TRAIN_ONLY_INPUTS.jsonl'),'prior_folds':sha(r/'next_step_train_diagnosis/FOLD_MANIFEST.json'),'prior_OOF':sha(r/'next_step_train_diagnosis/OOF_PREDICTIONS.jsonl')}
def features(rows,a,split):
 root=Path(a.root);identity=read(root/'natural/IDENTITY.json');prereg=read(root/'H2_PRE_REGISTRATION.json');assert identity['collector_sha256']==prereg['collector_sha256'];sources=[];out=[]
 for r in rows:
  assert r['split']==split and not r['is_demonstration'];f=root/'natural'/(split+'_'+hashlib.sha256(r['id'].encode()).hexdigest()+'.json');h=read(f);assert h['split']==split and h['id']==r['id'] and h['source_identity']==identity;assert h['input_ids_sha256']==r['input_ids_sha256'];assert h['native_source_sha256']==r['capture_sha256']
  for fam in ['candidate','unrelated']:
   assert h['features'][fam][0] is None
   assert np.asarray(h['features'][fam][1:]).shape==(3,16) and np.isfinite(np.asarray(h['features'][fam][1:])).all()
  out.append({**r,'latent':h['features']['candidate'],'unrelated':h['features']['unrelated']});sources.append({'id':r['id'],'split':split,'path':str(f),'sha256':sha(f)})
 return out,sources,identity
def predictions(m,rows,setting,family,seed,model,fold=None):
 x,y,ids,depths=m.array(rows,setting,family,seed);pp=m.probs(model,x);b,l=m.losses(pp,y);out=[]
 for i,key in enumerate(ids):out.append({'id':key,'setting':setting,'family':family,'depth':depths[i],'outer_fold':fold,'utility_class':int(y[i]),'probabilities':pp[i].tolist(),'utility_prediction':int(pp[i].argmax()),'brier':float(b[i]),'log_loss':float(l[i])})
 return out,m.metrics(pp,y)
def paired(m,pred,old=None):
 result=[]
 for setting in ['retrospective','prospective']:
  for family in ['output','unrelated','shuffled']+(['old_geometry'] if old is not None else []):
   pp=[p for p in pred if p['setting']==setting and p['family']=='combined'];rr=[p for p in (old if family=='old_geometry' else pred) if p['setting']==setting and p['family']==('combined' if family=='old_geometry' else family)];pmap={(p['id'],p['depth']):p for p in pp};rmap={(p['id'],p['depth']):p for p in rr};assert set(pmap)==set(rmap);byid=collections.defaultdict(list)
   for key in sorted(pmap):
    assert pmap[key]['utility_class']==rmap[key]['utility_class'];byid[key[0]].append(rmap[key]['brier']-pmap[key]['brier'])
   ids=sorted(byid);delta=[float(np.mean(byid[k])) for k in ids];result.append({'setting':setting,'comparator':family,'n_questions':len(ids),'Brier_improvement_comparator_minus_H2_combined':float(np.mean(delta)),'paired_question_bootstrap95':m.bootstrap(delta),'resampling':'10000 whole-question draws; fixed lexical ID order; seed20261113','per_question_improvement':[{'id':k,'delta':d} for k,d in zip(ids,delta)]})
 return result
def actions(m,rows,pred):
 summary=[];individual=[]
 for setting in ['retrospective','prospective']:
  for family in ['output','combined','unrelated','shuffled']:
   cache={(p['id'],p['depth']):p['utility_prediction'] for p in pred if p['setting']==setting and p['family']==family};deltas=[];loops=[]
   for r in rows:
    d=3 if setting=='retrospective' and cache[(r['id'],4)]==2 else 4
    if setting=='prospective':
     for k in [2,3]:
      if cache[(r['id'],k)] in [2,3]:d=k;break
    ok=r['pred'][d-1]==r['gold'];base=r['pred'][3]==r['gold'];delta=int(ok)-int(base);deltas.append(delta);loops.append(d if setting=='prospective' else 4);individual.append({'id':r['id'],'split':r['split'],'setting':setting,'family':family,'prediction':r['pred'][d-1],'gold':r['gold'],'selected_depth':d,'replay_loops':loops[-1],'correct':ok,'native_K4_correct':base,'delta_correct':delta,'permission':'SUPERVISED_DIAGNOSTIC_FIXED_MAPPING','actual_online_latency':'NOT_RUN'})
   order=sorted(range(len(rows)),key=lambda i:rows[i]['id']);ordered=[deltas[i] for i in order];rs=[r for r in individual if r['setting']==setting and r['family']==family];summary.append({'setting':setting,'family':family,'n':len(rows),'correct':sum(r['correct'] for r in rs),'native_K4_correct':sum(r['native_K4_correct'] for r in rs),'net_correct':sum(deltas),'rescue':sum(d==1 for d in deltas),'damage':sum(d==-1 for d in deltas),'mean_replay_loops':float(np.mean(loops)),'paired_question95_net_accuracy':m.bootstrap(ordered),'scope':'Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free.'})
 return summary,individual
def alpha_scores(m,rows,setting,family,outer=None):
 scores=[]
 for alpha in [.1,1.,10.]:
  total=0.;n=0
  for inner in range(3):
   ti=[r for r in rows if r['inner_fold']!=inner];vi=[r for r in rows if r['inner_fold']==inner];seed_outer=0 if outer is None else outer;xt,yt,_,_=m.array(ti,setting,family,1000+seed_outer*10+inner);xv,yv,_,_=m.array(vi,setting,family,2000+seed_outer*10+inner);model=m.fit(xt,yt,alpha);loss,_=m.losses(m.probs(model,xv),yv);total+=loss.sum();n+=len(loss)
  scores.append((float(total/n),alpha))
 return scores
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['source-freeze','train','validation'],required=True);p.add_argument('--root',default='/root/autodl-tmp/ouro-e7');p.add_argument('--e6',default='/root/autodl-tmp/ouro-e6a');p.add_argument('--out',default='/root/autodl-tmp/ouro-e7/analysis');p.add_argument('--validation-fixture');a=p.parse_args();root=Path(a.root);e6=Path(a.e6);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);h=hashes(a);sf=out/'H2_ANALYSIS_SOURCE_FREEZE.json'
 if a.phase=='source-freeze':
  assert not sf.exists();write(sf,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hashes':h,'permission':'SUPERVISED_DIAGNOSTIC, not training-free or a deployed method','primary_G1':'Prospective H2-combined 4class Brier must beat output, H2 unrelated, H2 shuffle; all three lexical-question bootstrap95 lower bounds >0','multiple_comparison':'Three simultaneous necessary gates; no substitute endpoint or posthoc choice','same_baseline':'Original class-balanced ridge/softmaxT1/10dim outputs/.1,1,10 alpha/5outer3inner groups/exact original seeds and row order; output OOF checked against prior','loop1':'None unavailable; arrays only d2+','final_train_model':'Each setting/family alpha by3 Train inner folds, seeds1000+inner and2000+inner; full Train array seed3000; Val seed4000. Final coeff and all choices frozen before Val read.','G2_actions':read(root/'H2_PRE_REGISTRATION.json')['frozen_actions'],'Val':'Once after Train-only policy/model freeze; previously exposed299, not independent confirmation. No Test path or file loaded.','outcomes_seen_before_source_freeze':False,'new_GPU_forwards':0});print('H2_SOURCE_FROZEN',sha(sf));return
 assert read(sf)['hashes']==h
 m=load_base(e6)
 if a.phase=='train':
  assert read(root/'natural/COMPLETE.json')['status']=='COMPLETE';assert not (out/'H2_TRAIN_FROZEN_SPEC.json').exists();t=time.perf_counter();rows0=jsonl(e6/'next_step_train_diagnosis/TRAIN_ONLY_INPUTS.jsonl');assert len(rows0)==1094 and all(r['split']=='train' for r in rows0);rows,sources,identity=features(rows0,a,'train');assert sha(e6/'next_step_train_diagnosis/FOLD_MANIFEST.json')==read(root/'H2_PRE_REGISTRATION.json')['prior_train_folds_sha256'];write(out/'FOLD_MANIFEST.json',[{k:r[k] for k in ['id','group','outer_fold','inner_fold','input_ids_sha256']} for r in rows]);pred=[];summary=[];alphas=[];coeff={};finalcoeff={};finalspec=[]
  for setting in ['retrospective','prospective']:
   for family in ['output','combined','unrelated','shuffled']:
    subset=[]
    for outer in range(5):
     tr=[r for r in rows if r['outer_fold']!=outer];va=[r for r in rows if r['outer_fold']==outer];scores=alpha_scores(m,tr,setting,family,outer);alpha=min(scores)[1];xt,yt,_,_=m.array(tr,setting,family,3000+outer);model=m.fit(xt,yt,alpha);pr,met=predictions(m,va,setting,family,4000+outer,model,outer);subset.extend(pr);alphas.append({'setting':setting,'family':family,'outer_fold':outer,'selected_alpha':alpha,'inner_scores':scores})
     for key,value in zip(['mean','scale','coef'],model):coeff[f'{setting}_{family}_{outer}_{key}']=value
    pred.extend(subset);pp=np.asarray([r['probabilities'] for r in subset]);yy=np.asarray([r['utility_class'] for r in subset]);summary.append({'setting':setting,'family':family,**m.metrics(pp,yy)});scores=alpha_scores(m,rows,setting,family);alpha=min(scores)[1];xt,yt,_,_=m.array(rows,setting,family,3000);model=m.fit(xt,yt,alpha);finalspec.append({'setting':setting,'family':family,'alpha':alpha,'inner_scores':scores,'train_shuffle_seed':3000,'validation_shuffle_seed':4000})
    for key,value in zip(['mean','scale','coef'],model):finalcoeff[f'{setting}_{family}_{key}']=value
    print('H2_OOF_DONE',setting,family,summary[-1],flush=True)
  old=jsonl(e6/'next_step_train_diagnosis/OOF_PREDICTIONS.jsonl');omap={(r['id'],r['setting'],r['depth']):r for r in old if r['family']=='output'};new=[r for r in pred if r['family']=='output'];assert len(new)==len(omap)
  maxerr=0.
  for r in new:
   ref=omap[(r['id'],r['setting'],r['depth'])];assert r['outer_fold']==ref['outer_fold'] and r['utility_class']==ref['utility_class'];err=float(np.max(np.abs(np.array(r['probabilities'])-ref['probabilities'])));maxerr=max(maxerr,err);assert np.allclose(r['probabilities'],ref['probabilities'],rtol=1e-12,atol=1e-14)
  putrows(out/'H2_OOF_PREDICTIONS.jsonl',pred);write(out/'H2_OOF_SUMMARY.json',summary);write(out/'H2_NESTED_ALPHA_SELECTION.json',alphas);np.savez_compressed(out/'H2_OUTER_MODELS.npz',**coeff);np.savez_compressed(out/'H2_FINAL_TRAIN_MODELS.npz',**finalcoeff);comparisons=paired(m,pred,old);write(out/'H2_PAIRED_COMPARISONS.json',comparisons);g1rows=[r for r in comparisons if r['setting']=='prospective' and r['comparator'] in ['output','unrelated','shuffled']];assert len(g1rows)==3;g1=all(r['paired_question_bootstrap95'][0]>0 for r in g1rows);act,ind=actions(m,rows,pred);write(out/'H2_FIXED_ACTION_DIAGNOSTICS.json',act);putrows(out/'H2_OOF_ACTIONS.jsonl',ind);putrows(out/'H2_TRAIN_FEATURES.jsonl',[{'id':r['id'],'split':'train','features':{'candidate':r['latent'],'unrelated':r['unrelated']},'input_ids_sha256':r['input_ids_sha256']} for r in rows]);write(out/'H2_OUTPUT_BASELINE_REPRODUCTION_QA.json',{'status':'PASS','output_transition_records':len(new),'maximum_probability_difference':maxerr,'source':str(e6/'next_step_train_diagnosis/OOF_PREDICTIONS.jsonl')})
  freeze={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hashes':h,'source_freeze_sha256':sha(sf),'capture_identity':identity,'train_capture_sources':sources,'final_models_sha256':sha(out/'H2_FINAL_TRAIN_MODELS.npz'),'outer_models_sha256':sha(out/'H2_OUTER_MODELS.npz'),'final_model_specs':finalspec,'G1_pass':g1,'G1_rule':'All3 prospective comparator-minus-H2 Brier bootstrap95 lower bounds >0','G2_permission':'SUPERVISED branch eligible for root online consideration only if G1 passes; OOF actions always diagnostic','data_scope':'Train1094 only; zero Val/Test in fitting or selection; historical ARC mechanism/Test exposure remains.','new_GPU_forwards':0,'cpu_wall_seconds':time.perf_counter()-t};write(out/'H2_TRAIN_FROZEN_SPEC.json',freeze);write(out/'H2_TRAIN_COMPLETE.json',{'status':'COMPLETE','n':1094,'OOF_records':len(pred),'G1_pass':g1,'freeze_sha256':sha(out/'H2_TRAIN_FROZEN_SPEC.json'),'cpu_wall_seconds':time.perf_counter()-t,'new_GPU_forwards':0});report(out,summary,comparisons,g1,act,'Train OOF');print('H2_TRAIN_COMPLETE',g1,flush=True)
 else:
  assert a.validation_fixture;freeze=read(out/'H2_TRAIN_FROZEN_SPEC.json');assert freeze['hashes']==h and sha(out/'H2_FINAL_TRAIN_MODELS.npz')==freeze['final_models_sha256'];assert not (out/'H2_VALIDATION_COMPLETE.json').exists();rows0=jsonl(a.validation_fixture);assert len(rows0)==299 and all(r['split']=='validation' for r in rows0);rows,sources,identity=features(rows0,a,'validation');assert identity==freeze['capture_identity'];coeff=np.load(out/'H2_FINAL_TRAIN_MODELS.npz');pred=[];summary=[]
  for s in freeze['final_model_specs']:
   setting,family=s['setting'],s['family'];model=tuple(coeff[f'{setting}_{family}_{k}'] for k in ['mean','scale','coef']);pr,met=predictions(m,rows,setting,family,s['validation_shuffle_seed'],model);pred.extend(pr);summary.append({'setting':setting,'family':family,**met})
  comparisons=paired(m,pred);act,ind=actions(m,rows,pred);putrows(out/'H2_VALIDATION_PREDICTIONS.jsonl',pred);putrows(out/'H2_VALIDATION_ACTIONS.jsonl',ind);write(out/'H2_VALIDATION_SUMMARY.json',summary);write(out/'H2_VALIDATION_PAIRED_COMPARISONS.json',comparisons);write(out/'H2_VALIDATION_FIXED_ACTION_DIAGNOSTICS.json',act);write(out/'H2_VALIDATION_COMPLETE.json',{'status':'COMPLETE','n':299,'train_freeze_sha256':sha(out/'H2_TRAIN_FROZEN_SPEC.json'),'validation_fixture_sha256':sha(a.validation_fixture),'sources':sources,'new_fits':0,'new_parameter_selections':0,'new_GPU_forwards':0,'scope':'Once exposed Validation review after Train freeze; not independent confirmation; no Test'});report(out,summary,comparisons,freeze['G1_pass'],act,'Exposed Validation review');print('H2_VALIDATION_COMPLETE',flush=True)
def report(out,summary,comparisons,g1,act,scope):
 text='# H2 '+scope+'\n\nAll analysis/implementation: Sol; Astra provided the separately frozen scientific directive. SUPERVISED_DIAGNOSTIC, not training-free or a deployable method. No Test examples, no new GPU forwards, no Ouro parameter changes.\n\nPrimary G1 Train prospective four-class Brier: '+('PASS' if g1 else 'FAIL')+'. G1 requires positive lower confidence limits against all three prescribed controls. Secondary endpoints do not change this decision.\n\n| Timing | Family | Brier | Logloss | Macro-F1 | ECE10 |\n|---|---|---:|---:|---:|---:|\n'
 for r in summary:text+=f"| {r['setting']} | {r['family']} | {r['brier']:.6f} | {r['log_loss']:.6f} | {r['macro_f1']:.4f} | {r['ECE10']:.4f} |\n"
 text+='\n## Paired proper-loss contrasts\n\nPositive means H2 combined improves proper Brier loss. All depths stay within the question; lexical IDs fix bootstrap draw alignment;10000 samples, seed20261113. Intervals are conditional on the correlated cross-validation fits, not independent training-seed uncertainty.\n\n| Timing | Comparator | Improvement | 95% CI |\n|---|---|---:|---|\n'
 for r in comparisons:
  lo,hi=r['paired_question_bootstrap95'];text+=f"| {r['setting']} | {r['comparator']} | {r['Brier_improvement_comparator_minus_H2_combined']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |\n"
 text+='\n## Fixed supervised class-to-action replay\n\nRetrospective completedK4 class2 selects K3; otherwise K4, with cost4. Prospective class2 or3 stops currentK2/K3; otherwise continues. No threshold/strategy search. OOF/held-out replay depths are not measured latency.\n\n'+json.dumps(act,indent=2)+'\n\n## Interpretation and scope\n\nThe 16-dimensional H2 block is candidate-aligned natural lens geometry, replacing the previous16 geometry coordinates. It is never inserted into old index11/index5 gates. d1=None is unavailable, not a zero measurement; all observations use d2+. Original10 LP-output features and original class-balanced ridge/T1/seeds/folds are unchanged. Output-only OOF equality is an explicit technical control. Extra-feature families have26 dimensions; output-only10 is not falsely called equal parameter count. No base-model weights were trained.\n\nE3 Train1030/Validation299 guided earlier mechanism selection and historical Test was repeatedly exposed. Current Train-only group cross-validation and frozen exposed-Val review do not create independent benchmark confirmation. A G1 pass permits consideration of a supervised inference experiment, not a claim that the practical research goal or a deployable accuracy/compute improvement has been achieved. A G1 failure leaves future direction to the separately authorized Astra scientific directive; neither primary metric nor mapping is changed after results.\n'
 (out/('H2_TRAIN_REPORT.md' if scope=='Train OOF' else 'H2_VALIDATION_REPORT.md')).write_text(text)
if __name__=='__main__':main()
