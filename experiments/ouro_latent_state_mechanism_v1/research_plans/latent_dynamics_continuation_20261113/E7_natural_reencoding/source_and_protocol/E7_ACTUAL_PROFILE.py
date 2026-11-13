"""Conditional true E7 runtime measurement. prepare is CPU/input-only; root authorizes GPU phases."""
import argparse,json,hashlib,time,datetime,os,importlib.util
from pathlib import Path
import numpy as np
MODEL_SHA='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
SOURCE_SHA='c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413'
SEED='E7-runtime128-20261113:'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open(encoding='utf8') as f:
  for l in f:
   if l.strip():yield json.loads(l)
def default(v):
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,np.ndarray):return v.tolist()
 raise TypeError(type(v).__name__)
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.partial');t.write_text(json.dumps(x,indent=2,allow_nan=False,default=default)+'\n');os.replace(t,p)
def load_methods(p):
 s=importlib.util.spec_from_file_location('original_frozen_methods',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class StopAtPrefix(Exception):pass
def prefix_decision(prefix,amod):
 # Same original row-wise FP64 centering/probabilities/entropy; no future score rows.
 x=np.asarray(prefix,dtype=np.float64);x=x-x.mean(axis=1,keepdims=True);prob=amod.probabilities(x);d=len(x);prediction=x.argmax(1);entropy=-(prob[-1]*np.log(np.maximum(prob[-1],1e-300))).sum()/np.log(x.shape[1]);return d==4 or (d in [2,3] and prediction[-1]==prediction[-2] and prob[-1].max()>=.6 and entropy<=.5)
def prepare(a):
 root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);freeze=out/'PROFILE_SAMPLE_AND_SOURCE_FREEZE.json';assert not freeze.exists();meta={}
 for q in lines(a.metadata):
  assert q['split']=='train' and not q['is_demonstration'];meta[q['id']]={'group':q['group'],'input_ids_sha256':q['input_ids_sha256']}
 assert len(meta)==1094
 eligible={}
 for q in lines(a.inputs):
  assert q['split'] in ['train','validation']
  if q['split']=='train' and q['p']==q['s']==0 and len(q['targets'])==4:
   assert q['id'] in meta and meta[q['id']]['input_ids_sha256']==q['input_ids_sha256'];eligible[q['id']]=meta[q['id']]
 assert len(eligible)==1092;representative={}
 for key,v in eligible.items():representative[v['group']]=min(key,representative.get(v['group'],key))
 ordered=sorted(representative,key=lambda g:(hashlib.sha256((SEED+g).encode()).hexdigest(),representative[g]));groups=ordered[:128];ids=[representative[g] for g in groups];assert len(ids)==128 and len(set(groups))==128;views={key:{} for key in ids}
 for q in lines(a.inputs):
  if q['split']=='train' and q['id'] in views and q['s']==0 and q['p'] in [0,2]:
   assert not any(k in q for k in ['gold','query_gold','correct','correct_choice']) and len(q['targets'])==4;views[q['id']][q['p']]=q
 fixture=[]
 for group,key in zip(groups,ids):
  assert set(views[key])=={0,2};fixture.append({'id':key,'group':group,'selection_hash':hashlib.sha256((SEED+group).encode()).hexdigest(),'identity':views[key][0],'pair_view':views[key][2]})
 file=out/'PROFILE_GOLDFREE_TRAIN128.jsonl';file.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in fixture),encoding='utf8');md=Path(a.model)
 write(freeze,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PREPARED_ONLY_NO_GPU','profile_source_sha256':sha(__file__),'fixture_sha256':sha(file),'inputs_sha256':sha(a.inputs),'old_Train_metadata_sha256':sha(a.metadata),'old_metadata_fields_used_only':['id','group','input_ids_sha256'],'no_gold_in_runtime_fixture':True,'sample_rule':'fourchoiceTrain only; fixed prefix+original question group SHA ordering; one lexical ID representative/group; first128 groups; independent of any new outcome','selection_seed_prefix':SEED,'sample_ids':ids,'sample_groups':groups,'technical_first_two_ids':ids[:2],'runtime_original_methods_sha256':sha(a.methods),'frozen_analysis_source_sha256':sha(root/'E7_REENCODING_ANALYSIS.py'),'E7_protocol_sha256':sha(root/'E7_PROTOCOL_FREEZE.json'),'tokenizer_sha256':sha(md/'tokenizer.json'),'model_source_sha256':sha(md/'modeling_ouro.py'),'model_weights_expected_sha256':MODEL_SHA,'launch_gate':'Only after Train decision freeze: pair_profile_eligible or selected_gate_threshold is not None; otherwise NOT_RUN before importing torch or acquiring GPU. Root alone launches. Eligible GPU phases also require completed exposedVal review and full22280 capture.','policies':'Native4 and original strong cal_early_.6 entropy.5, plus eligible fixedpair and/or eligible frozen nativegate; no new methods/thresholds','order':'Cyclic question-index rotation of this fixed policy list; retain first cold unit and firstquestion separately','timing':'Each policy actually tokenizes its one/two goldfree prompt(s), asserts IDs, tensors to CUDA, performs all required forwards/readouts/CPU decisions/FP64 aggregation and synchronizes. Model/readout/decision and full end-to-end windows separate. Pure hash/cache QA outside windows; scalar counter hooks and R-only QA clones remain inside and are disclosed.','permission':'Budget measurement only, no goal-success declaration, no model fit/Test data, no GPU action by prepare phase.'});print('E7_PROFILE_PREPARED',sha(freeze),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','technical','profile'],required=True);p.add_argument('--root',default='/root/autodl-tmp/ouro-e7/reencoding');p.add_argument('--out');p.add_argument('--inputs');p.add_argument('--metadata',default='/root/autodl-tmp/ouro-e6a/next_step_train_diagnosis/TRAIN_ONLY_INPUTS.jsonl');p.add_argument('--model',default='/root/autodl-tmp/ouro-e4-e5/model');p.add_argument('--methods',default='/root/autodl-tmp/ouro-e6a/E6A/E6A_METHODS.py');a=p.parse_args();root=Path(a.root);a.out=a.out or str(root/'actual_profile');a.inputs=a.inputs or str(root/'E7_VARIANTS.jsonl');out=Path(a.out)
 if a.phase=='prepare':prepare(a);return
 sf=out/'PROFILE_SAMPLE_AND_SOURCE_FREEZE.json';frozen=read(sf);assert frozen['profile_source_sha256']==sha(__file__) and frozen['runtime_original_methods_sha256']==sha(a.methods);fixturefile=out/'PROFILE_GOLDFREE_TRAIN128.jsonl';assert frozen['fixture_sha256']==sha(fixturefile);trainfile=root/'analysis/E7_TRAIN_FROZEN_DECISIONS.json';assert trainfile.exists(),'Await frozen Train decisions; no GPU action';decision=read(trainfile);assert decision['source_hashes']['analysis_source']==frozen['frozen_analysis_source_sha256'];pair=decision['pair_profile_eligible'];gate_threshold=decision['selected_gate_threshold'];assert isinstance(pair,bool) and (gate_threshold is None or gate_threshold in [.25,.5,.75,1.])
 if not pair and gate_threshold is None:
  write(out/'PROFILE_NOT_RUN.json',{'status':'NOT_RUN','reason':'Neither frozen Train practical-pair nor nativegate profile criterion passed','Train_decision_sha256':sha(trainfile),'sample_and_source_freeze_sha256':sha(sf),'new_GPU_forwards':0,'torch_imported':False});print('E7_PROFILE_NOT_RUN_NO_GPU');return
 valfile=root/'analysis/VALIDATION_COMPLETE.json';assert valfile.exists() and read(root/'full_capture/COMPLETE.json')['n']==22280,'Root must review complete Train/Val conclusions before any permitted profile';assert read(valfile)['train_decisions_sha256']==sha(trainfile) and read(valfile)['source_sha256']==frozen['frozen_analysis_source_sha256']
 phaseout=out/a.phase;phaseout.mkdir(exist_ok=True);assert not (phaseout/'COMPLETE.json').exists(),'Never silently repeat completed runtime profile'
 fixture=list(lines(fixturefile));assert len(fixture)==128 and [q['id'] for q in fixture]==frozen['sample_ids'];policies=['native_K4','strong_cal_early_0.6']+(['fixed_pair_K2_K2'] if pair else [])+(['native_gate_selected'] if gate_threshold is not None else [])
 permission={'source_sha256':sha(__file__),'sample_freeze_sha256':sha(sf),'Train_decision_sha256':sha(trainfile),'Val_completion_sha256':sha(valfile),'selected_gate_threshold':gate_threshold,'pair_eligible':pair,'policies':policies}
 if a.phase=='profile':
  tech=read(out/'technical/COMPLETE.json');assert tech['status']=='PASS' and tech['permission']==permission and tech['question_ids']==frozen['technical_first_two_ids'],'First two fixed technical questions must pass before full128'
 else:fixture=fixture[:2]
 # Lock before torch/CUDA import; never steal, wait for, terminate or relaunch another task.
 import fcntl
 lock=open('/root/autodl-tmp/ouro-e6a/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import_start=time.perf_counter();import torch,transformers
 transformers.utils.import_utils._torchvision_available=False
 from transformers import AutoModelForCausalLM
 from tokenizers import Tokenizer
 import_seconds=time.perf_counter()-import_start;torch.set_grad_enabled(False);torch.set_num_threads(1);torch.manual_seed(2021);torch.backends.cuda.matmul.allow_tf32=False;validation_start=time.perf_counter();assert sha(Path(a.model)/'model.safetensors')==MODEL_SHA and sha(Path(a.model)/'modeling_ouro.py')==SOURCE_SHA and frozen['tokenizer_sha256']==sha(Path(a.model)/'tokenizer.json');file_validation_seconds=time.perf_counter()-validation_start
 t=time.perf_counter();tok=Tokenizer.from_file(str(Path(a.model)/'tokenizer.json'));tokenizer_load_seconds=time.perf_counter()-t;t=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;torch.cuda.synchronize();model_load_seconds=time.perf_counter()-t
 import inspect
 assert sha(inspect.getfile(model.__class__))==SOURCE_SHA and model.early_exit_step is None and model.early_exit_threshold==1.;amod=load_methods(a.methods);versions=[(n,id(v),v._version) for n,v in model.named_parameters()];version_schema=[(n,list(v.shape),str(v.dtype),v._version) for n,v in model.named_parameters()];write(phaseout/'PARAMETER_VERSION_SCHEMA.json',version_schema)
 counter={'body_forwards':0,'first_layer_calls':0,'head_calls':0,'gate_calls':0};current_shape=None
 def body_hook(m,args,kwargs):counter['body_forwards']+=1
 def layer_hook(m,args,kwargs):counter['first_layer_calls']+=1
 def head_hook(m,args,result):counter['head_calls']+=1;assert result.ndim==3 and tuple(result.shape[:2])==current_shape and result.shape[2]==model.config.vocab_size
 def gate_hook_count(m,args,result):counter['gate_calls']+=1
 hooks=[model.model.register_forward_pre_hook(body_hook,with_kwargs=True),model.model.layers[0].register_forward_pre_hook(layer_hook,with_kwargs=True),model.lm_head.register_forward_hook(head_hook),model.model.early_exit_gate.register_forward_hook(gate_hook_count)]
 def th(x):return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
 def set_depth(d):model.model.total_ut_steps=d;model.config.total_ut_steps=d
 def readout(z,view):
  # LP and candidate transfer are algorithm work; R clone is bounded QA instrumentation, timed and disclosed.
  r=z[0,view['R']].detach().clone();scores=r.float().log_softmax(-1)[view['targets']].cpu().numpy();return r,scores
 def run_policy(q,policy):
  nonlocal current_shape
  required=[q['identity'],q['pair_view']] if policy=='fixed_pair_K2_K2' else [q['identity']];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();ct=dict(counter);e2estart=time.perf_counter();tensors=[];tokentime=time.perf_counter()
  for view in required:
   ids=tok.encode(view['prompt'],add_special_tokens=False).ids;assert ids==view['ids'] and hashlib.sha256(json.dumps(ids).encode()).hexdigest()==view['input_ids_sha256'];tensors.append(torch.tensor([ids],device='cuda'))
  torch.cuda.synchronize();tokenization_tensor_seconds=time.perf_counter()-tokentime;modelstart=time.perf_counter();qa=[];prefix=[];cdf_prefix=[];endpoint={};actual_depths=[]
  if policy in ['native_K4','fixed_pair_K2_K2']:
   target_depth=2 if policy=='fixed_pair_K2_K2' else 4;sem=[]
   for view,ids in zip(required,tensors):
    current_shape=tuple(ids.shape);set_depth(target_depth);full=model(ids,labels=None,use_cache=False,exit_at_step=target_depth-1,logits_to_keep=0).logits;z,lp=readout(full,view);qa.append({'variant_id':view['variant_id'],'depth':target_depth,'R_logits':z});actual_depths.append(target_depth);sem.append(lp.astype(np.float64)[np.argsort(view['display_to_original'])]);del full
   if policy=='fixed_pair_K2_K2':score=((sem[0]-sem[0].mean())+(sem[1]-sem[1].mean()))/2;endpoint={'prediction_semantic':int(score.argmax()),'semantic_aggregate_scores':score.tolist(),'semantic_top_set':np.flatnonzero(score==score.max()).tolist()}
   else:endpoint={'prediction_semantic':int(sem[0].argmax()),'semantic_scores':sem[0].tolist()}
  elif policy=='strong_cal_early_0.6':
   view=required[0];ids=tensors[0];current_shape=tuple(ids.shape);set_depth(4)
   def norm_callback(m,args,h):
    full=model.lm_head(h);z,lp=readout(full,view);d=len(prefix)+1;prefix.append(lp.tolist());qa.append({'variant_id':view['variant_id'],'depth':d,'R_logits':z})
    if prefix_decision(prefix,amod):endpoint.update(prediction_semantic=view['display_to_original'][int(lp.argmax())],selected_depth=d);raise StopAtPrefix()
   handle=model.model.norm.register_forward_hook(norm_callback)
   try:
    try:model.model(ids,use_cache=False,past_key_values=None)
    except StopAtPrefix:pass
   finally:handle.remove()
   assert 'selected_depth' in endpoint;actual_depths.append(endpoint['selected_depth'])
  elif policy=='native_gate_selected':
   view=required[0];ids=tensors[0];current_shape=tuple(ids.shape);set_depth(4);pdf=[];remaining=None;saved={}
   def gate_callback(m,args,result):
    nonlocal remaining
    d=len(pdf)+1;hazard=torch.sigmoid(result.squeeze(-1))
    if remaining is None:remaining=torch.ones_like(hazard)
    if d<4:mass=hazard*remaining;remaining=remaining*(1.0-hazard)
    else:mass=remaining
    pdf.append(mass);padded=pdf+[torch.zeros_like(mass) for _ in range(4-d)];cdf=torch.cumsum(torch.stack(padded,dim=2),dim=2);cdf_prefix.append(cdf[0,view['R'],:d].detach().clone())
    if d==4 or bool(cdf[0,view['R'],d-1]>=gate_threshold):saved.update(hidden=args[0],depth=d);raise StopAtPrefix()
   handle=model.model.early_exit_gate.register_forward_hook(gate_callback)
   try:
    try:model.model(ids,use_cache=False,past_key_values=None)
    except StopAtPrefix:pass
   finally:handle.remove()
   assert 'depth' in saved;full=model.lm_head(saved['hidden']);z,lp=readout(full,view);qa.append({'variant_id':view['variant_id'],'depth':saved['depth'],'R_logits':z});actual_depths.append(saved['depth']);endpoint={'prediction_semantic':view['display_to_original'][int(lp.argmax())],'selected_depth':saved['depth']};del full
  else:raise ValueError(policy)
  torch.cuda.synchronize();end=time.perf_counter();counts={k:counter[k]-ct[k] for k in counter};return {'policy':policy,'prediction':endpoint,'actual_depths':actual_depths,'actual_counts':counts,'end_to_end_seconds':end-e2estart,'model_readout_decision_seconds':end-modelstart,'tokenization_ID_assertion_tensor_sync_seconds':tokenization_tensor_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()},qa,cdf_prefix
 def unit(view):
  file=root/'full_capture'/('train_'+hashlib.sha256(view['variant_id'].encode()).hexdigest()+'.json');u=read(file);assert u['status']=='COMPLETE' and u['input_ids_sha256']==view['input_ids_sha256'];assert u['unit_payload_sha256']==canonical({k:v for k,v in u.items() if k!='unit_payload_sha256'});assert u['identity']['model_sha256']==MODEL_SHA and u['identity']['model_source_sha256']==SOURCE_SHA;return u
 rows=[];started=time.perf_counter()
 try:
  for i,q in enumerate(fixture):
   assert q['identity']['split']==q['pair_view']['split']=='train' and len(q['identity']['targets'])==4 and not any(k in q for k in ['gold','query_gold']);assert all(not any(k in view for k in ['gold','query_gold','correct','correct_choice']) for view in [q['identity'],q['pair_view']]);order=policies[i%len(policies):]+policies[:i%len(policies)]
   for orderindex,policy in enumerate(order):
    result,qa,cdfprefix=run_policy(q,policy)
    # All pure hash/cache/golden-rule comparisons occur after both reported timers.
    cached={};checks=[]
    for endpoint in qa:
     view=q['identity'] if endpoint['variant_id']==q['identity']['variant_id'] else q['pair_view'];u=cached.setdefault(endpoint['variant_id'],unit(view));d=endpoint['depth'];h=th(endpoint['R_logits']);assert h==u['native'][d-1]['full_logits_sha256'];checks.append({'variant_id':endpoint['variant_id'],'depth':d,'R_full_vocabulary_sha256':h,'native_exact':True})
    identity=cached.get(q['identity']['variant_id']) or unit(q['identity']);counts=result['actual_counts'];depths=result['actual_depths'];assert counts['first_layer_calls']==sum(depths)
    if policy=='native_K4':assert counts=={'body_forwards':1,'first_layer_calls':4,'head_calls':1,'gate_calls':4};assert result['prediction']['prediction_semantic']==identity['native'][3]['prediction_semantic']
    elif policy=='fixed_pair_K2_K2':
     assert counts=={'body_forwards':2,'first_layer_calls':4,'head_calls':2,'gate_calls':4};v2=cached[q['pair_view']['variant_id']];s1=np.asarray(identity['native'][1]['scores'],dtype=np.float64)[np.argsort(identity['display_to_original'])];s2=np.asarray(v2['native'][1]['scores'],dtype=np.float64)[np.argsort(v2['display_to_original'])];v=((s1-s1.mean())+(s2-s2.mean()))/2;assert np.array_equal(v,np.asarray(result['prediction']['semantic_aggregate_scores'])) and int(v.argmax())==result['prediction']['prediction_semantic']
    elif policy=='strong_cal_early_0.6':
     expected=amod.infer([n['scores'] for n in identity['native']],amod.calibration_grid()['cal_early_0.6']);d=depths[0];assert counts=={'body_forwards':1,'first_layer_calls':d,'head_calls':d,'gate_calls':d-1};assert expected['selected_depth']==d and identity['display_to_original'][expected['prediction']]==result['prediction']['prediction_semantic']
    elif policy=='native_gate_selected':
     d=depths[0];assert counts=={'body_forwards':1,'first_layer_calls':d,'head_calls':1,'gate_calls':d};expected=identity['gate_R']['threshold_readouts'][str(gate_threshold)];assert d==expected['selected_depth'] and result['prediction']['prediction_semantic']==expected['prediction_semantic']
     for k,c in enumerate(cdfprefix,1):assert c.float().cpu().tolist()==identity['gate_R']['exit_cdf'][:k],(q['id'],k,'prefix CDF differs from BF16 full4-column public arithmetic')
    assert versions==[(n,id(v),v._version) for n,v in model.named_parameters()];result.update(id=q['id'],group=q['group'],sample_index=i,policy_order=orderindex,cold_first_unit=i==0 and orderindex==0,first_question_units=i==0,endpoint_QA=checks,gate_CDF_prefix_exact=True if policy=='native_gate_selected' else None,weights_version_unchanged=True,QA_outside_timers=True);rows.append(result)
    with (phaseout/'rows.jsonl').open('a') as f:f.write(json.dumps(result,default=default,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    del qa,cdfprefix
   print('E7_PROFILE_QA_PASS',a.phase,i+1,len(fixture),q['id'],flush=True)
  summary={}
  for policy in policies:
   rr=[r for r in rows if r['policy']==policy];warm=[r for r in rr if not r['first_question_units']];summary[policy]={'n':len(rr),'actual_counts_sum':{k:sum(r['actual_counts'][k] for r in rr) for k in counter},'mean_end_to_end_seconds':np.mean([r['end_to_end_seconds'] for r in rr]),'median_end_to_end_seconds':np.median([r['end_to_end_seconds'] for r in rr]),'mean_model_readout_decision_seconds':np.mean([r['model_readout_decision_seconds'] for r in rr]),'mean_end_to_end_excluding_first_question':np.mean([r['end_to_end_seconds'] for r in warm]) if warm else None,'max_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rr)}
  write(phaseout/'COMPLETE.json',{'status':'PASS','permission':permission,'question_ids':[q['id'] for q in fixture],'n_questions':len(fixture),'n_policy_units':len(rows),'source_sha256':sha(__file__),'fixture_sha256':sha(fixturefile),'model_sha256':MODEL_SHA,'loaded_source_sha256':SOURCE_SHA,'torch':torch.__version__,'transformers':transformers.__version__,'GPU':torch.cuda.get_device_name(),'tokenizer_load_seconds':tokenizer_load_seconds,'model_load_seconds':model_load_seconds,'framework_import_seconds':import_seconds,'file_validation_seconds':file_validation_seconds,'profile_wall_seconds_excluding_setup':time.perf_counter()-started,'all_actual_counts':counter,'policy_summary':summary,'weights_modified':False,'scope':'Actual control flow and end-to-end tokenization/tensor/forward/readout/CPU decision/aggregate synchronization. Small R-only QA clones and scalar hooks counted in timers; pure hashes/cache comparisons outside. No timing subtraction of required algorithm work. Per-policy peak includes resident model and bounded R clone instrumentation. No gold in worker or runtime fixture; no Test; no claim Goal attained.'});print('E7_ACTUAL_PROFILE_COMPLETE',a.phase,len(rows),flush=True)
 finally:
  for h in hooks:h.remove()
if __name__=='__main__':main()
