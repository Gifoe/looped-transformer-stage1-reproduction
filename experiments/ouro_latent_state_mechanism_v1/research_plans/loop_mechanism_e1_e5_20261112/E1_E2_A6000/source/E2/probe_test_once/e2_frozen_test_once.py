"""One-shot E2 frozen test evaluator. Run only upon Astra authorization."""
import os,json,gzip,hashlib,datetime,pathlib,math
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
if not __debug__:raise RuntimeError('optimized Python forbidden')
import numpy as np
from scipy.special import logsumexp
from safetensors import safe_open
B=pathlib.Path('/home/luansiyu/loop-latent/repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112');E=B/'E2';P=E/'probe_fit_attempt3';OUT=E/'probe_test_once';S=E/'E2_fixed_probe_states.safetensors';R=E/'E2_latent_transition_rows.jsonl.gz';A=E/'E2_posthoc_audit.json';O=B/'orders/E2_PROBE_FIT_ORDER_v3.md';F=E/'e2_fit_probe_v3.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={'selection':(P/'train_val_selection.json','e7b8823c6c29358077f357340252f298955b2087764b83fd89cf7a571fbe6759'),'models':(P/'selected_models.npz','204a3788301c7aa826a198b9e40ce495ef81284b6cd41a4758dfd942f8650890'),'states':(S,'8ccae86df9786e88fa63219d10c9ba101ae96f49793da7f63ae01449e49e62e5'),'rows':(R,'5d749790c56ab7bf2de3a9a64609e0a5b383fe3b7c5b2ec0cc5d5129586daac4'),'audit':(A,'f9b90e4cd875de601b3b3df539aa4c34c2b9b425b77078cee441660cf275ce49'),'order':(O,'ff9463fca57a3c6b680d6052e53eed0f43b1c4140e02c9393fc616763c11704b'),'fit':(F,'ca24cef4aab75616d3267b9a4479ea508e2fc4e8b8e78d119bc06d3983e3652c')}
for name,(p,h) in files.items():assert sha(p)==h,name
OUT.mkdir(exist_ok=False)
def write_exclusive(path,obj):
 with path.open('x',encoding='utf8') as f:json.dump(obj,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
start={'status':'STARTED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'frozen_hashes':{k:v[1] for k,v in files.items()},'script_sha256':sha(pathlib.Path(__file__)),'test_exposure':'Prior ARC outcomes and test candidate OOV inventory were already observed; no independent blind test claim.'};write_exclusive(OUT/'STARTED.json',start)
try:
 meta=json.loads(files['selection'][0].read_text());models=np.load(files['models'][0],allow_pickle=False);classes=meta['classes_target_ids'];ci={v:i for i,v in enumerate(classes)};assert classes==[330,340,389,414,422] and len(meta['models'])==16
 with gzip.open(R,'rt') as f:rows=[json.loads(line) for line in f]
 train=[r for r in rows if r['split']=='train' and not r['demo']];test=[r for r in rows if r['split']=='test'];assert len(train)==1094 and len(test)==1172 and len({r['id'] for r in test})==1172
 counts=np.ones(len(classes));byn={}
 for r in train:
  ix=ci[r['targets'][r['gold']]];counts[ix]+=1;byn.setdefault(len(r['targets']),np.ones(len(classes)))[ix]+=1
 def score(r,values,probs,kind):
  values=np.array(values,dtype=np.float64);probs=np.array(probs,dtype=np.float64);g=r['gold'];wrong=[j for j in range(len(values)) if j!=g]; k1=r['native_scores'][0];fixed=max(wrong,key=lambda j:k1[j]);top=values.max();ties=np.flatnonzero(values==top).tolist();pred=ties[0];return {'id':r['id'],'targets':r['targets'],'gold_index':g,'group':r['group'],'natural_no_top_tie':r['natural_no_top_tie'],'candidate_values':values.tolist(),'candidate_probabilities':probs.tolist(),'prediction_index':pred,'top_ties':ties,'correct':pred==g,'gold_allwrong_gap':float(values[g]-values[wrong].mean()),'fixed_wrong_index':fixed,'fixed_wrong_gap':float(values[g]-values[fixed]),'dynamic_gap':float(values[g]-values[wrong].max()),'cross_entropy':float(logsumexp(values)-values[g]),'brier':float(sum((v-(i==g))**2 for i,v in enumerate(probs))),'value_type':kind}
 def summary(rows):
  if not rows:return {'n':0}
  confidence=np.array([max(x['candidate_probabilities']) for x in rows]);correct=np.array([x['correct'] for x in rows],float);bins=np.minimum(np.searchsorted(np.linspace(0,1,11),confidence,side='right')-1,9);assert (bins>=0).all() and (bins<=9).all();hist=np.bincount(bins,minlength=10);assert hist.sum()==len(rows)
  ece=sum(hist[i]/len(rows)*abs(correct[bins==i].mean()-confidence[bins==i].mean()) for i in range(10) if hist[i]);out={'n':len(rows),'accuracy':float(correct.mean()),'conditional_cross_entropy':float(np.mean([x['cross_entropy'] for x in rows])),'brier':float(np.mean([x['brier'] for x in rows])),'ece_10':float(ece),'ece_bin_counts':hist.tolist(),'top_tie_count':sum(len(x['top_ties'])>1 for x in rows)}
  for k in ('gold_allwrong_gap','fixed_wrong_gap','dynamic_gap'):out['mean_'+k]=float(np.mean([x[k] for x in rows]))
  return out
 allres={};groups=('rescue','damage','stable_correct','stable_wrong')
 def aggregate(key,items):
  good=[x for x in items if x.get('status','OK')=='OK'];allres[key]={'all':summary(good),'oov_count':len(items)-len(good),'groups':{g:summary([x for x in good if x['group']==g]) for g in groups},'natural_no_top_tie':summary([x for x in good if x['natural_no_top_tie']])}
 with (OUT/'per_question.jsonl').open('x',encoding='utf8') as output, safe_open(str(S),framework='pt',device='cpu') as store:
  for site in ('l18_input','post_norm'):
   for typ in ('raw','unit_l2'):
    for u in (0,1):
     X=np.stack([store.get_tensor(r['split']+'_'+hashlib.sha256(r['id'].encode()).hexdigest()+f'_u{u}_'+site).float().numpy().ravel().astype(np.float64) for r in test]);assert X.shape==(1172,2048) and np.isfinite(X).all()
     if typ=='unit_l2':
      norms=np.linalg.norm(X,axis=1,keepdims=True);X=np.divide(X,norms,out=np.zeros_like(X),where=norms!=0)
     for trainloop in (1,2):
      for control in ('real','permuted'):
       name=f'{site}_{typ}_trainK{trainloop}_{control}';mu=models[name+'_mu'];std=models[name+'_std'];w=models[name+'_w'];T=np.column_stack(((X-mu)/std,np.ones(len(X))));logits=T@w;items=[]
       for i,r in enumerate(test):
        if any(t not in ci for t in r['targets']):items.append({'id':r['id'],'status':'UNSUPPORTED_CANDIDATE_OOV','targets':r['targets']});continue
        values=logits[i,[ci[t] for t in r['targets']]];probs=np.exp(values-logsumexp(values));items.append(score(r,values,probs,'probe_logits'))
       key=f'{name}_evalK{u+1}';aggregate(key,items)
       for item in items:output.write(json.dumps({'model':key,**item},allow_nan=False)+'\n')
  for u in (0,1):
   items=[]
   for r in test:
    values=np.array(r['native_scores'][u],float);items.append(score(r,values,np.exp(values-logsumexp(values)),'native_full_vocab_logprob_conditionalized_over_candidates'))
   key=f'native_K{u+1}';aggregate(key,items)
   for item in items:output.write(json.dumps({'model':key,**item},allow_nan=False)+'\n')
  for mode in ('uniform','global','candidate_count'):
   items=[]
   for r in test:
    if any(t not in ci for t in r['targets']) or (mode=='candidate_count' and len(r['targets']) not in byn):items.append({'id':r['id'],'status':'UNSUPPORTED_CANDIDATE_OOV_OR_COUNT'});continue
    cs=np.ones(len(r['targets'])) if mode=='uniform' else (byn[len(r['targets'])] if mode=='candidate_count' else counts)[[ci[t] for t in r['targets']]];p=cs/cs.sum();items.append(score(r,np.log(p),p,'train_only_prior_log_probability'))
   aggregate('prior_'+mode,items)
   for item in items:output.write(json.dumps({'model':'prior_'+mode,**item},allow_nan=False)+'\n')
  output.flush();os.fsync(output.fileno())
 write_exclusive(OUT/'summary.json',{'scope':'Frozen decodability, not natural use or causality. Native source values are full-vocabulary log-probabilities; probabilities here are conditionalized over provided candidates. Prior ARC test exposure and candidate OOV inventory occurred.','results':allres})
 for name,(p,h) in files.items():assert sha(p)==h,name
 write_exclusive(OUT/'SUCCESS.json',{'status':'COMPLETE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'test_n':len(test),'models':16,'model_by_loop_results':32,'frozen_hashes':{k:v[1] for k,v in files.items()},'script_sha256':sha(pathlib.Path(__file__)),'question_sha256':sha(OUT/'per_question.jsonl'),'summary_sha256':sha(OUT/'summary.json')})
except BaseException as e:
 write_exclusive(OUT/'FAILED.json',{'status':'FAILED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exception_type':type(e).__name__,'message':str(e),'test_predictions_may_have_been_computed':True});raise

