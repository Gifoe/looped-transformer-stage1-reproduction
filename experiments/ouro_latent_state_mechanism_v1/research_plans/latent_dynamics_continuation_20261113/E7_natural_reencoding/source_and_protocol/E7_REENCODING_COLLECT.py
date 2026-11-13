"""Natural, gold-free E7 reencoding capture. GPU invocation belongs to coordinator."""
import argparse,json,hashlib,time,datetime,os,fcntl,inspect
from pathlib import Path

MODEL_SHA='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
SOURCE_SHA='c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413'
GATE_SHAS={'weight':'fa92644802e800123d4be3d7f18e08fe47e27ac89e6c15a4308cdd568479accc','bias':'ab2bc9800ff2d8b79cf751d7f6200ffb21b0ee75ae0d91e3017556e25cca8a3c'}
THRESHOLDS=[.25,.5,.75,1.]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def input_rows(p):
 with Path(p).open(encoding='utf8') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial');tmp.write_text(json.dumps(x,allow_nan=False));os.replace(tmp,p)
def read(p):return json.loads(Path(p).read_text())
def validate_input(q):
 assert q['split'] in ['train','validation'] and not any(k in q for k in ['gold','query_gold','correct','correct_choice'])
 assert q['R']==len(q['ids'])-1 and q['input_ids_sha256']==hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()
 m=len(q['targets']);assert m in [3,4,5] and len(q['display_to_original'])==m and sorted(q['display_to_original'])==list(range(m));assert len(q['labels'])==m and len(set(q['labels']))==m and len(set(q['targets']))==m
 assert 0<=q['p']<m and 0<=q['s']<4 and q['variant_id']==q['id']+'__p'+str(q['p'])+'_s'+str(q['s'])
 assert q['display_to_original']==[(j+q['p'])%m for j in range(m)]
 assert isinstance(q['generator_sha256'],str) and len(q['generator_sha256'])==64
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);p.add_argument('--model',required=True);p.add_argument('--native-root',required=True);p.add_argument('--technical',action='store_true');a=p.parse_args()
 # No labels or Test paths are accepted. Input file ordering is coordinator-frozen.
 seen=set();n=0;inputsha=sha(a.inputs)
 for q in input_rows(a.inputs):
  validate_input(q);key=(q['split'],q['variant_id']);assert key not in seen;seen.add(key);n+=1
 assert n>0
 import torch,transformers
 transformers.utils.import_utils._torchvision_available=False
 from transformers import AutoModelForCausalLM
 torch.set_grad_enabled(False);torch.set_num_threads(1);torch.manual_seed(2021);torch.backends.cuda.matmul.allow_tf32=False
 lock=open('/root/autodl-tmp/ouro-e6a/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 wall=time.perf_counter();modeldir=Path(a.model);assert sha(modeldir/'model.safetensors')==MODEL_SHA;assert sha(modeldir/'modeling_ouro.py')==SOURCE_SHA
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);runid=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+str(os.getpid());runpath=out/'runs'/(runid+'.json');journal=out/'runs'/(runid+'.jsonl');journal.parent.mkdir(exist_ok=True)
 model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False
 assert sha(inspect.getfile(model.__class__))==SOURCE_SHA;assert sha(inspect.getfile(model.config.__class__))==sha(modeldir/'configuration_ouro.py');assert model.early_exit_step is None and model.early_exit_threshold==1.0
 def th(x):return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
 loadedgates={name:th(getattr(model.model.early_exit_gate,name)) for name in GATE_SHAS};assert loadedgates==GATE_SHAS
 core={'collector_sha256':sha(__file__),'model_sha256':MODEL_SHA,'model_source_sha256':SOURCE_SHA,'loaded_model_source_sha256':sha(inspect.getfile(model.__class__)),'config_sha256':sha(modeldir/'config.json'),'configuration_sha256':sha(modeldir/'configuration_ouro.py'),'loaded_configuration_sha256':sha(inspect.getfile(model.config.__class__)),'loaded_gate_parameter_sha256':loadedgates,'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),'dtype':'bfloat16','attention':'sdpa','use_cache':False,'eval':not model.training,'loop_count':4,'head_shape':'full [batch,sequence,vocabulary] at every native depth','score_protocol':'Joint option label FP32 full-vocabulary log_softmax, candidate order=display','gate_thresholds':THRESHOLDS,'gate_readout_scope':'Offline full-body R threshold readout; not true early termination','tie_rules':{'production':'first displayed candidate index among exactly maximal original BF16 candidate logits','sensitivity':'minimum original semantic index among exact BF16 top set'},'native_root':str(Path(a.native_root).resolve())}
 ip=out/'IDENTITY.json'
 if ip.exists():assert read(ip)==core,'Source/model/runtime namespace identity mismatch; use a distinct output namespace'
 else:write(ip,core)
 versions=[(n,id(v),v._version) for n,v in model.named_parameters()];version_schema=[(n,list(v.shape),str(v.dtype),v._version) for n,v in model.named_parameters()];version_sha=canonical(version_schema)
 counter={'model_body_forwards':0,'head_calls':0,'first_layer_calls':0};phase={};new=0;reused=0;preserved=0;production_seconds=0.;technical_seconds=0.;retained=[]
 def body_count(m,args,kwargs):counter['model_body_forwards']+=1
 def first_count(m,args,kwargs):counter['first_layer_calls']+=1
 def head_count(m,args,result):
  counter['head_calls']+=1
  assert result.ndim==3 and result.shape[:2]==current_shape and result.shape[2]==model.config.vocab_size,'Full-shaped head is mandatory'
 handles=[model.model.register_forward_pre_hook(body_count,with_kwargs=True),model.model.layers[0].register_forward_pre_hook(first_count,with_kwargs=True),model.lm_head.register_forward_hook(head_count)]
 def snapshot():return dict(counter)
 def diff(old):return {k:counter[k]-old[k] for k in counter}
 def log(event,**kwargs):
  with journal.open('a') as f:f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':event,**kwargs})+'\n');f.flush();os.fsync(f.fileno())
 def describe(v,lp,mapping):
  # Candidate logits retain exact representable BF16 values despite JSON float serialization.
  top=(v==v.max()).nonzero().flatten().cpu().tolist();display=int(v.argmax());sem=[mapping[i] for i in top]
  return {'candidate_logits':v.float().cpu().tolist(),'scores':lp.cpu().tolist(),'prediction':display,'prediction_display':display,'prediction_semantic':mapping[display],'top_tie':len(top)>1,'top_display_indices':top,'top_semantic_indices':sorted(sem),'semantic_index_tiebreak_prediction':min(sem)}
 def gate_pdf(gates):
  remaining=torch.ones_like(gates[0].squeeze(-1));pdf=[];hazards=[]
  for idx,g in enumerate(gates):
   lam=torch.sigmoid(g.squeeze(-1));hazards.append(lam)
   if idx<len(gates)-1:p=lam*remaining;remaining=remaining*(1.0-lam)
   else:p=remaining
   pdf.append(p)
  pdf=torch.stack(pdf,dim=2);return hazards,pdf,torch.cumsum(pdf,dim=2)
 try:
  for i,q in enumerate(input_rows(a.inputs)):
   key=q['split']+'_'+hashlib.sha256(q['variant_id'].encode()).hexdigest();file=out/(key+'.json');qsha=canonical(q)
   if file.exists():
    try:r=read(file)
    except (ValueError,OSError):
     backup=file.with_name(file.name+'.incomplete_preserved_'+runid);os.replace(file,backup);preserved+=1;log('PRESERVED_INCOMPLETE',path=str(backup));r=None
    if r is not None:
     assert r.get('status')=='COMPLETE' and r['identity']==core and r['variant_row_sha256']==qsha and r['input_ids_sha256']==q['input_ids_sha256'],'Completed unit identity mismatch; never overwrite'
     assert r['unit_payload_sha256']==canonical({k:v for k,v in r.items() if k!='unit_payload_sha256'}),'Completed unit checksum mismatch'
     assert len(r['native'])==4 and len(r['gate_R']['exit_cdf'])==4 and r['weight_versions_unchanged']
     if a.technical:assert r['technical']['status']=='PASS','Technical reuse requires a prior complete technical gate'
     reused+=1;retained.append({'file':str(file),'sha256':sha(file),'variant_id':q['variant_id'],'reused_this_invocation':True});log('REUSED',variant_id=q['variant_id'],file=str(file));continue
   log('START',variant_id=q['variant_id'],input_sha256=q['input_ids_sha256']);current_shape=(1,len(q['ids']));ids=torch.tensor([q['ids']],device='cuda');R=q['R'];mapping=q['display_to_original'];model.model.total_ut_steps=4;model.config.total_ut_steps=4
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();before=snapshot();_,hidden,gates=model.model(ids,use_cache=False,past_key_values=None);assert len(hidden)==len(gates)==4
   native=[];Rlogits=[]
   for depth,h in enumerate(hidden,1):
    full=model.lm_head(h);z=full[0,R].detach().clone();v=z[q['targets']];scores=z.float().log_softmax(-1)[q['targets']];row=describe(v,scores,mapping);row.update(depth=depth,full_logits_sha256=th(z),candidate_logits_dtype=str(v.dtype),scores_dtype=str(scores.dtype));native.append(row);Rlogits.append(z);del full
   hazards,pdf,cdf=gate_pdf(gates);readouts={}
   for threshold in THRESHOLDS:
    mask=cdf[0,R]>=threshold;step=int(mask.float().argmax()) if bool(mask.any()) else 3;readouts[str(threshold)]={'selected_depth':step+1,**native[step],'executed_loops':4,'body_cost_kind':'full_body_offline_readout'}
   gate={'logits':[float(g[0,R,0]) for g in gates],'hazards':[float(g[0,R]) for g in hazards],'exit_pdf':pdf[0,R].float().cpu().tolist(),'exit_cdf':cdf[0,R].float().cpu().tolist(),'logits_dtype':str(gates[0].dtype),'hazard_dtype':str(hazards[0].dtype),'PDF_dtype':str(pdf.dtype),'CDF_dtype':str(cdf.dtype),'PDF_R_sha256':th(pdf[0,R]),'CDF_R_sha256':th(cdf[0,R]),'final_PDF_is_remaining_mass':True,'threshold_readouts':readouts}
   replica={};identity_match=None
   if q['p']==q['s']==0:
    oldfile=Path(a.native_root)/q['split']/(q['split']+'_'+hashlib.sha256(q['id'].encode()).hexdigest()+'.json');old=read(oldfile);assert old['input_ids_sha256']==q['original_input_ids_sha256']==q['input_ids_sha256'];assert old['targets']==q['targets'];assert old['identity']['model_sha256']==MODEL_SHA and old['identity']['gpu']==core['gpu']
    assert [n['full_logits_sha256'] for n in old['native']]==[n['full_logits_sha256'] for n in native];assert [n['scores'] for n in old['native']]==[n['scores'] for n in native]
    early,final=Rlogits[0],Rlogits[3];top=final.float().softmax(-1).topk(2).values;strength=1-(top[0]-top[1])
    for name,amount in [('loopcd_adaptive_cap1',strength),('loopcd_fixed_0.5',.5),('loopcd_cap0_identity',0.)]:
     adjusted=final.float()+amount*(final.float()-early.float());scores=adjusted.log_softmax(-1)[q['targets']];rr=describe(scores,scores,mapping);rr.pop('candidate_logits');rr.update(strength=float(amount),reference_depth=1,executed_loops=4,uses_full_vocabulary_confidence=True,tie_basis='legacy E6 FP32 candidate log-probability exact equality');replica[name]=rr
     if name=='loopcd_cap0_identity':assert rr['scores']==native[3]['scores']
     if name in old.get('replica_baselines',{}):assert rr['scores']==old['replica_baselines'][name]['scores'] and rr['prediction']==old['replica_baselines'][name]['prediction'] and rr['top_tie']==old['replica_baselines'][name]['top_tie']
    identity_match={'status':'PASS','E6_source_file':str(oldfile),'E6_source_sha256':sha(oldfile),'four_native_full_vocabulary_R_hashes_exact':True,'input_tokens_sha256_exact':True,'LoopCD_existing_control_scores_exact':True}
   torch.cuda.synchronize();prod=time.perf_counter()-start;prodcounts=diff(before);assert prodcounts=={'model_body_forwards':1,'head_calls':4,'first_layer_calls':4};production_seconds+=prod;log('PRODUCTION_COMPLETE',variant_id=q['variant_id'],counts=prodcounts,seconds=prod)
   tech={'status':'NOT_REQUESTED'};techstart=time.perf_counter();techbefore=snapshot()
   if a.technical:
    checks=[]
    for depth in range(1,5):
     model.model.total_ut_steps=depth;model.config.total_ut_steps=depth;cb=snapshot();full=model(ids,labels=None,use_cache=False,exit_at_step=depth-1,logits_to_keep=0).logits;assert th(full[0,R])==native[depth-1]['full_logits_sha256'];counts=diff(cb);assert counts=={'model_body_forwards':1,'head_calls':1,'first_layer_calls':depth};checks.append({'depth':depth,'status':'PASS','full_shaped_head':True,'R_full_vocabulary_sha256':th(full[0,R]),'actual_counts':counts});del full
    model.model.total_ut_steps=4;model.config.total_ut_steps=4
    # Exact exported default readout, per token and full head shape, from production body states.
    threshold_mask=cdf>=model.early_exit_threshold;steps=threshold_mask.float().argmax(dim=2);steps[~threshold_mask.any(dim=2)]=3;stacked=torch.stack(hidden,dim=2);idx=steps.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,1,stacked.size(-1));selected=torch.gather(stacked,2,idx).squeeze(2);cb=snapshot();expected=model.lm_head(selected);expected_counts=diff(cb);cb=snapshot();actual=model(ids,labels=None,use_cache=False,exit_at_step=None,exit_threshold=None,logits_to_keep=0).logits;stock_counts=diff(cb);assert expected_counts=={'model_body_forwards':0,'head_calls':1,'first_layer_calls':0};assert stock_counts=={'model_body_forwards':1,'head_calls':1,'first_layer_calls':4};assert torch.equal(expected,actual),'Full-sequence default readout differs from public forward';assert th(actual[0,R])==native[int(steps[0,R])]['full_logits_sha256'];default={'status':'PASS','stock_default_threshold':model.early_exit_threshold,'R_selected_depth':int(steps[0,R])+1,'all_sequence_logits_bitwise_equal':True,'full_sequence_logits_sha256':th(actual),'R_full_vocabulary_sha256':th(actual[0,R]),'manual_default_head_counts':expected_counts,'independent_stock_default_counts':stock_counts};del expected,actual,stacked,selected
    tech={'status':'PASS','independent_fixed_depths':checks,'stock_default':default}
   torch.cuda.synchronize();techseconds=time.perf_counter()-techstart;technical_seconds+=techseconds;techcounts=diff(techbefore)
   assert versions==[(n,id(v),v._version) for n,v in model.named_parameters()];row={'status':'COMPLETE','id':q['id'],'split':q['split'],'variant_id':q['variant_id'],'p':q['p'],'s':q['s'],'R':R,'token_count':len(q['ids']),'input_ids_sha256':q['input_ids_sha256'],'original_input_ids_sha256':q['original_input_ids_sha256'],'variant_row_sha256':qsha,'generator_sha256':q['generator_sha256'],'targets':q['targets'],'labels':q['labels'],'display_to_original':mapping,'original_to_display':[mapping.index(j) for j in range(len(mapping))],'identity':core,'native':native,'gate_R':gate,'identity_E6_bridge':identity_match,'replica_baselines':replica,'production_actual_counts':prodcounts,'production_physical_loop_applications':prodcounts['first_layer_calls'],'production_forward_scoring_seconds':prod,'technical':tech,'technical_actual_counts':techcounts,'technical_seconds':techseconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'weight_versions_unchanged':True,'parameter_version_schema_sha256':version_sha,'new_intervention':'none, input-only natural reencoding','storage':'scores, hashes and R gate scalars only; no large activations'};row['unit_payload_sha256']=canonical(row);write(file,row);new+=1;retained.append({'file':str(file),'sha256':sha(file),'variant_id':q['variant_id'],'reused_this_invocation':False});log('UNIT_COMPLETE',variant_id=q['variant_id'],file=str(file),sha256=sha(file),actual_counts=diff(before));print('E7_CAPTURE',i+1,n,q['split'],q['variant_id'],round(prod,4),flush=True);del hidden,gates,Rlogits,ids,pdf,cdf,hazards
  assert sha(a.inputs)==inputsha,'Input manifest mutated during capture';assert versions==[(n,id(v),v._version) for n,v in model.named_parameters()];receipt={'status':'COMPLETE','run_id':runid,'n':n,'new_completed_views':new,'reused_completed_views':reused,'incomplete_files_preserved':preserved,'this_invocation_actual_counts':counter,'production_seconds_new_only':production_seconds,'technical_seconds_new_only':technical_seconds,'wall_seconds_including_model_load':time.perf_counter()-wall,'weights_modified':False,'identity':core,'inputs_file_sha256':inputsha,'inputs_file':str(Path(a.inputs).resolve()),'technical_requested':a.technical,'records':retained,'resume_scope':'Only atomic complete unit JSON with matching core/source/model/canonical variant input identity. Unrecorded interrupted work is not inferred as zero; START/phase journal retained.'};write(runpath,receipt);write(out/'COMPLETE.json',receipt);print('E7_CAPTURE_COMPLETE',new,reused,flush=True)
 except BaseException as ex:
  write(runpath,{'status':'FAILED_OR_INTERRUPTED','run_id':runid,'exception':repr(ex),'new_completed_views':new,'reused_completed_views':reused,'this_invocation_actual_counts':counter,'inputs_file_sha256':inputsha,'identity':core,'wall_seconds':time.perf_counter()-wall});raise
 finally:
  for h in handles:h.remove()
if __name__=='__main__':main()
