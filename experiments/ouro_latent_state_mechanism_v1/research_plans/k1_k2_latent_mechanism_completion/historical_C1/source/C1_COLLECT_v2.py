"""C1 fresh source-bound references and independent L18-branch treatments; root alone launches GPU."""
import argparse,json,hashlib,time,datetime,os,inspect,fcntl,gzip
from pathlib import Path
import torch,transformers
from safetensors.torch import save as st_save,load as st_load
from branch_interfaces_v2 import attention_reference_operator,actual_L18_branch_interface,q9_region,bits,thash,UndefinedControl
MODEL_SHA='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af';MODEL_SOURCE='c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413'
CONFIG_SHA='ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3';CONFIGURATION_SHA='950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def name(q):return hashlib.sha256(q['id'].encode()).hexdigest()
def tensor_bytes(path):
 path=Path(path)
 return path.read_bytes() if path.exists() else gzip.decompress(Path(str(path)+'.gz').read_bytes())
def tensor_sha(path):return hashlib.sha256(tensor_bytes(path)).hexdigest()
def load_tensor(path):return st_load(tensor_bytes(path))
def save(path,tensors,root):
 path=Path(str(path)+'.gz');path.parent.mkdir(parents=True,exist_ok=True);values={k:v.detach().cpu().contiguous().clone() for k,v in tensors.items()};raw=st_save(values);back=st_load(raw);assert set(back)==set(values) and all(bits(back[k],v) for k,v in values.items());data=gzip.compress(raw,compresslevel=6,mtime=0);assert gzip.decompress(data)==raw;tmp=path.with_name(path.name+'.partial');tmp.write_bytes(data);os.replace(tmp,path);return {'tensor_sha256':hashlib.sha256(raw).hexdigest(),'tensor_archive_sha256':sha(path),'tensor_storage':'gzip','tensor_file':str(path.relative_to(root))}
def score(z,targets):
 assert z.ndim==1 and len(z)==49152 and torch.isfinite(z).all();lp=z.float().log_softmax(-1);c=z[targets];top=(c==c.max()).nonzero().flatten().cpu().tolist();h=thash(z);return {'Rlogits_sha256':h,'full_logits_sha256':h,'candidate_logits':[float(x) for x in c],'scores':[float(x) for x in lp[targets]],'logsumexp_fp32':float(z.float().logsumexp(-1)),'prediction':int(c.argmax()),'top_tie':len(top)>1,'top_ties':top,'logits_dtype':str(z.dtype),'score_dtype':'torch.float32'}
def receipt(rec):
 out={}
 for k,v in rec.items():
  if torch.is_tensor(v):out[k]={'sha256':thash(v),'shape':list(v.shape),'dtype':str(v.dtype)}
  elif k!='technical_only_full_qkv':out[k]=v
 return out
FIELD_MAP={'x_R':'x_actual_R','a_R':'a_actual_R','O_R':'O_actual_R','Q_R':'Q_actual_R','K_R':'K_actual_R','V_R':'V_actual_R','y_R':'y_branch_R','y_before_R':'y_actual_before_branch_R','b_R':'b_actual_R','after_mlp_R':'after_mlp_R','final_norm_R':'final_norm_R','MLP_input_R':'MLP_input_R','MLP_skip_R':'MLP_skip_shared_alias_R'}
def pack(prefix,rec,full=True):
 keys=FIELD_MAP if full else {'x_branch_R':'x_branch_used_R','a_branch_R':'a_branch_used_R','y_R':'y_branch_R','b_R':'b_actual_R','after_mlp_R':'after_mlp_R','final_norm_R':'final_norm_R','MLP_input_R':'MLP_input_R','MLP_skip_R':'MLP_skip_shared_alias_R'}
 return {prefix+k:rec[v] for k,v in keys.items()}
def sites_pack(prefix,sites):
 result={}
 for i,v in sites.items():
  for key in ['before','after','exact_before_cast','cast_residual','normalization_scale_fp64']:
   if key in v:result[f'{prefix}{i:02d}_{key}']=v[key]
 return result
def forward(model,q,*,p=False,deltas=None,norm=False,x=None,a=None,expected_Q=None,rewrite=False,K1_head=False):
 assert model.model.total_ut_steps==2;assert q['R']==len(q['ids'])-1
 ids=torch.tensor([q['ids']],device='cuda');before=[(n,id(v),v._version) for n,v in model.named_parameters()];counts={'body_forwards':0,'first_layer_calls':0,'head_calls':0};handles=[]
 def body(m,args,kwargs):counts['body_forwards']+=1
 def layer(m,args,kwargs):counts['first_layer_calls']+=1
 def head(m,args,result):counts['head_calls']+=1;assert result.shape==(1,len(q['ids']),49152)
 handles=[model.model.register_forward_pre_hook(body,with_kwargs=True),model.model.layers[0].register_forward_pre_hook(layer,with_kwargs=True),model.lm_head.register_forward_hook(head)];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
 try:
  with q9_region(model,R=q['R'],prefix_len=q['prefix_len'],sequence_length=len(q['ids']),perturbed=p,expected_donor=expected_Q) as qr,attention_reference_operator(model,q['R'],sequence_length=len(q['ids']),reference_delta=deltas,sign=1 if p else -1,norm_preserve=norm) as sites,actual_L18_branch_interface(model,q['R'],sequence_length=len(q['ids']),x_branch=x,a_branch=a,rewrite_identity=rewrite) as branch:
   _,states,gates=model.model(ids,use_cache=False,past_key_values=None);assert len(states)==2;z=model.lm_head(states[1])[0,q['R']].detach().clone();k1=model.lm_head(states[0])[0,q['R']].detach().clone() if K1_head else None
 finally:
  for h in handles:h.remove()
 torch.cuda.synchronize();elapsed=time.perf_counter()-t;assert counts=={'body_forwards':1,'first_layer_calls':2,'head_calls':2 if K1_head else 1};assert before==[(n,id(v),v._version) for n,v in model.named_parameters()];row=score(z,q['targets']);row.update(actual_counts=counts,forward_window_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),weights_version_unchanged=True);return row,z,k1,qr,sites,branch
def paths(root,split,key):
 p=root/'references'/split/(key+'.json');return p,p.with_suffix('.safetensors')
def getref(root,split,key,identity=None):
 j,t=paths(root,split,key);r=read(j);assert r['status']=='COMPLETE' and tensor_sha(t)==r['tensor_sha256'] and sha(Path(str(t)+'.gz'))==r['tensor_archive_sha256'];assert identity is None or r['source_identity']==identity;return r,load_tensor(t)
def nonR(a,b):
 for k in ['non_R_x_sha256','non_R_Q_sha256','non_R_K_sha256','non_R_V_sha256','non_R_O_sha256','non_R_after_mlp_sha256','non_R_final_norm_sha256']:assert a[k]==b[k],k
def preserve_actual(branch,reference):
 for key in ['x_R','a_R','O_R','Q_R','K_R','V_R']:assert bits(branch[FIELD_MAP[key]].cpu(),reference[key]),key
def references(model,qs,root,identity,native_root):
 reused=0;new=0;rows=[]
 for q in qs:
  j,tp=paths(root,q['split'],name(q))
  if j.exists():r,t=getref(root,q['split'],name(q),identity);assert r['input_row_sha256']==canonical(q);reused+=1;rows.append({'id':q['id'],'json_sha256':sha(j),'tensor_sha256':tensor_sha(tp),'new':False});continue
  nr,nz,k1,qr,na,nb=forward(model,q,K1_head=True);pr,pz,unused,pq,pa,pb=forward(model,q,p=True,expected_Q=qr['K1_Q']);d={i:na[i]['before'].double()-pa[i]['before'].double() for i in range(9,16)};ur,uz,unused,uq,us,ub=forward(model,q,deltas=d,expected_Q=qr['K1_Q']);nonR(receipt(nb),receipt(ub));t={'K1_logits':k1,'N_logits':nz,'P_logits':pz,'U_logits':uz,'Q9_K1_Q':qr['K1_Q'],**pack('N_',nb),**pack('P_',pb),**pack('U_',ub),**sites_pack('U_site_',us)}
  for i in d:t[f'N_a_{i:02d}']=na[i]['before'];t[f'P_a_{i:02d}']=pa[i]['before'];t[f'd_{i:02d}']=d[i]
  old=read(Path(native_root)/q['split']/(q['split']+'_'+name(q)+'.json'));assert old['input_ids_sha256']==q['input_ids_sha256'] and old['targets']==q['targets'] and old['identity']['gpu']==identity['gpu'];assert nr['Rlogits_sha256']==old['native'][1]['full_logits_sha256'] and thash(k1)==old['native'][0]['full_logits_sha256'];assert nr['scores']==old['native'][1]['scores'] and score(k1,q['targets'])['scores']==old['native'][0]['scores'];archive=save(tp,t,root);write(j,{'status':'COMPLETE','id':q['id'],'split':q['split'],'R':q['R'],'prefix_len':q['prefix_len'],'targets':q['targets'],'input_ids_sha256':q['input_ids_sha256'],'input_row_sha256':canonical(q),'source_identity':identity,**archive,'conditions':{'K1':score(k1,q['targets']),'N':nr,'P':pr,'U':ur},'N_receipt':receipt(nb),'P_receipt':receipt(pb),'U_receipt':receipt(ub),'N_Q9_receipt':receipt(qr),'P_Q9_receipt':receipt(pq),'U_Q9_receipt':receipt(uq),'native_same_hardware_bridge':'PASS_K1_K2_fullvocabR_and_FP32scores','dedup':'Reference branch tensors saved once; treatment actual-path hashes and exact checks bind shared operands to these raw tensors.'});new+=1;rows.append({'id':q['id'],'json_sha256':sha(j),'tensor_sha256':tensor_sha(tp),'new':True});print('C1_REFERENCE_COMPLETE',q['split'],q['id'],new,reused,flush=True)
 return {'new':new,'reused':reused,'records':rows}
def control(base,target,donor_delta,kind,sign):
 # All branch-control arithmetic runs on these archived CPU tensors in FP64, one writeback cast.
 b=base.double();t=target.double();delta=t-b;n0=b.norm();n1=t.norm();dose=delta.norm();scale=torch.ones((),dtype=torch.float64)
 if kind=='norm':
  if float(n0)==0 and float(n1)!=0:raise UndefinedControl('norm control zero base and nonzero target norm')
  scale=n1/n0 if float(n0)!=0 else scale;exact=b*scale
 else:
  direction=donor_delta.double();dn=direction.norm()
  if float(dose)==0:scale=torch.zeros((),dtype=torch.float64);exact=b.clone()
  elif float(dn)==0:raise UndefinedControl('nonzero requested dose but frozen donor direction is zero')
  else:scale=dose/dn;exact=b+sign*scale*direction
 after=exact.to(base.dtype);stats={'math_device':'CPU','math_dtype':'torch.float64','cast_dtype':str(base.dtype),'requested_delta_norm':float(dose),'actual_after_cast_delta_norm':float((after.double()-b).norm()),'base_norm':float(n0),'target_norm':float(n1),'after_norm':float(after.double().norm()),'scale_fp64':float(scale),'zero_requested_dose':float(dose)==0,'kind':kind,'sign':sign};return after,{'control_before':base,'control_exact_before_cast':exact,'control_after':after,'control_cast_residual':after.double()-exact,'control_scale_fp64':scale},stats
def treatments(model,qs,root,identity,donor_map,all_split_ids,technical,replay_ids):
 # Main scientific donor conditions require complete originalsplit refs before treatment batches.
 if not technical:
  for key in all_split_ids:assert paths(root,qs[0]['split'],hashlib.sha256(key.encode()).hexdigest())[0].exists(),'Complete split N/P/U reference stage is required before donor treatments'
 condition_names=read(root/'inputs/CONDITIONS.json');result=[]
 for q in qs:
  ref,t=getref(root,q['split'],name(q),identity);donor=donor_map[q['id']];dr,dt=getref(root,q['split'],hashlib.sha256(donor.encode()).hexdigest(),identity);d={i:t[f'd_{i:02d}'] for i in range(9,16)};conditions={'P_plus_L':{'p':True,'d':d},'UX':{'d':d,'x':t['N_x_R']},'UA':{'d':d,'a':t['N_a_R']},'UXA':{'d':d,'x':t['N_x_R'],'a':t['N_a_R']},'U_upstream_norm':{'d':d,'norm':True}};undefined={};control_tensors={};control_stats={}
  try:
   ud={}
   for i in range(9,16):
    target=t[f'd_{i:02d}'];v=dt[f'd_{i:02d}'];dn=v.norm();requested=target.norm()
    if float(requested)==0:ud[i]=torch.zeros_like(target)
    elif float(dn)==0:raise UndefinedControl('upstream frozen donor direction zero at physical layer'+str(i+1))
    else:ud[i]=requested*v/dn
   conditions['U_upstream_unrelated']={'d':ud};control_tensors['U_upstream_unrelated']={f'control_requested_delta_{i:02d}':v for i,v in ud.items()}
  except UndefinedControl as ex:undefined['U_upstream_unrelated']=str(ex)
  for branch,key in [('X','x_R'),('A','a_R')]:
   donor_delta=dt['N_'+key].double()-dt['U_'+key].double()
   for direction,base,target,sign in [('U',t['U_'+key],t['N_'+key],1),('N',t['N_'+key],t['U_'+key],-1)]:
    for kind in ['norm','unrelated']:
     cname=direction+'_'+branch+'_'+kind+('_rollback' if direction=='N' else '')
     try:
      value,ct,stats=control(base,target,donor_delta,kind,sign);conditions[cname]={**({'d':d} if direction=='U' else {}),('x' if branch=='X' else 'a'):value};control_tensors[cname]=ct;control_stats[cname]=stats
     except UndefinedControl as ex:undefined[cname]=str(ex)
  extras={}
  if technical or q['id'] in replay_ids:extras={'N_branch_identity':{'x':t['N_x_R'],'a':t['N_a_R'],'rewrite':True},'U_branch_identity':{'d':d,'x':t['U_x_R'],'a':t['U_a_R'],'rewrite':True},'NXU':{'x':t['U_x_R']},'NAU':{'a':t['U_a_R']},'NXUAU':{'x':t['U_x_R'],'a':t['U_a_R']}}
  path=root/'treatments'/q['split']/name(q);path.mkdir(parents=True,exist_ok=True);items={};ordered=[n for n in condition_names if n not in ['N','P','U']]+list(extras);conditions.update(extras)
  for cname in ordered:
   j=path/(cname+'.json');tp=j.with_suffix('.safetensors')
   if j.exists():old=read(j);assert old['source_identity']==identity and old['input_row_sha256']==canonical(q) and old['reference_json_sha256']==sha(paths(root,q['split'],name(q))[0]);assert old['status']=='UNAVAILABLE' or tensor_sha(tp)==old['tensor_sha256'];items[cname]={'status':old['status'],'json_sha256':sha(j)};continue
   if cname in undefined:write(j,{'status':'UNAVAILABLE','condition':cname,'id':q['id'],'split':q['split'],'reason':undefined[cname],'source_identity':identity,'input_row_sha256':canonical(q),'reference_json_sha256':sha(paths(root,q['split'],name(q))[0])});items[cname]={'status':'UNAVAILABLE','json_sha256':sha(j)};continue
   cfg=conditions[cname]
   try:row,z,unused,qr,sites,br=forward(model,q,p=cfg.get('p',False),deltas=cfg.get('d'),norm=cfg.get('norm',False),x=cfg.get('x'),a=cfg.get('a'),expected_Q=t['Q9_K1_Q'],rewrite=cfg.get('rewrite',False))
   except UndefinedControl as ex:
    write(j,{'status':'UNAVAILABLE','condition':cname,'id':q['id'],'split':q['split'],'reason':str(ex),'source_identity':identity,'input_row_sha256':canonical(q),'reference_json_sha256':sha(paths(root,q['split'],name(q))[0])});items[cname]={'status':'UNAVAILABLE','json_sha256':sha(j)};continue
   useU=cfg.get('d') is d and not cfg.get('norm',False) and not cfg.get('p',False);useN=cfg.get('d') is None and not cfg.get('p',False);background='U' if useU else 'N' if useN else None
   if background:
    preserve_actual(br,{key:t[background+'_'+key] for key in ['x_R','a_R','O_R','Q_R','K_R','V_R']});nonR(receipt(br),ref[background+'_receipt'])
   if cname in ['UXA','N_branch_identity']:assert bits(z.cpu(),t['N_logits'])
   if cname in ['U_branch_identity','NXUAU']:assert bits(z.cpu(),t['U_logits'])
   target={'NXU':'UA','NAU':'UX'}.get(cname)
   if target:targett=load_tensor(path/(target+'.safetensors'));assert bits(z.cpu(),targett['logits'])
   vectors={'logits':z,**pack('',br,full=background is None),**control_tensors.get(cname,{})}
   if cfg.get('d') is not None and not useU:vectors.update(sites_pack('site_',sites))
   if useU:
    for ii,sv in sites.items():
     for key in ['before','after','exact_before_cast','cast_residual','normalization_scale_fp64']:assert bits(sv[key].cpu(),t[f'U_site_{ii:02d}_{key}'])
   site_receipts={str(ii):receipt(sv) for ii,sv in sites.items()}
   archive=save(tp,vectors,root);write(j,{'status':'COMPLETE','id':q['id'],'split':q['split'],'condition':cname,'source_identity':identity,'input_ids_sha256':q['input_ids_sha256'],'input_row_sha256':canonical(q),'reference_json_sha256':sha(paths(root,q['split'],name(q))[0]),**archive,'actual_background_reference':background,'actual_L18_receipt':receipt(br),'upstream_site_receipts':site_receipts,'upstream_site_shared_reference':'U_site_' if useU else None,'Q9_receipt':receipt(qr),'control_metadata':control_stats.get(cname),'donor_id':donor if 'unrelated' in cname else q['id'],'donor_reference_sha256':sha(paths(root,q['split'],hashlib.sha256(donor.encode()).hexdigest())[0]) if 'unrelated' in cname else None,**row});items[cname]={'status':'COMPLETE','json_sha256':sha(j)};print('C1_TREATMENT_COMPLETE',q['split'],q['id'],cname,round(row['forward_window_seconds'],3),flush=True)
  result.append({'id':q['id'],'conditions':items,'reference_json_sha256':sha(paths(root,q['split'],name(q))[0])})
 return result
def aggregate_coverage(root,split,inputs,identity):
 conditions=[x for x in read(root/'inputs/CONDITIONS.json') if x not in ['N','P','U']];records=[];unavailable=[]
 for q in inputs:
  j,t=paths(root,split,name(q))
  if not j.exists():return
  ref=read(j);assert ref['status']=='COMPLETE' and ref['source_identity']==identity and sha(root/ref['tensor_file'])==ref['tensor_archive_sha256']
  for c in conditions:
   file=root/'treatments'/split/name(q)/(c+'.json')
   if not file.exists():return
   row=read(file);assert row['source_identity']==identity and row['reference_json_sha256']==sha(j) and row['input_row_sha256']==canonical(q)
   if row['status']=='UNAVAILABLE':unavailable.append({'id':q['id'],'condition':c,'reason':row['reason']})
   else:assert row['status']=='COMPLETE' and sha(root/row['tensor_file'])==row['tensor_archive_sha256']
   records.append({'id':q['id'],'condition':c,'status':row['status'],'json_sha256':sha(file)})
 write(root/(split.upper()+'_TREATMENTS_COMPLETE.json'),{'status':'COMPLETE','split':split,'n':len(inputs),'main_endpoint_slots':17*len(inputs),'K1_native_slots':len(inputs),'source_identity':identity,'unavailable_controls':unavailable,'treatment_records':records,'scope':'All fixed questions retained; unavailable control rows explicitly counted, no outcome-dependent exclusion.'})
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='/root/autodl-tmp/ouro-k1-k2-mechanism');p.add_argument('--stage',choices=['reference','treatment'],required=True);p.add_argument('--split',choices=['train','validation'],required=True);p.add_argument('--batch',choices=['all','technical_dependencies','technical2','train64','train1030','independent_replay'],default='all');p.add_argument('--model',default='/root/autodl-tmp/ouro-e4-e5/model');p.add_argument('--native-root',default='/root/autodl-tmp/ouro-e6a/joint');p.add_argument('--technical',action='store_true');a=p.parse_args();r=Path(a.root);inputs=[json.loads(l) for l in (r/'inputs'/(a.split+'.jsonl')).read_text().splitlines()];assert len(inputs)==(1094 if a.split=='train' else 299) and all(q['split']==a.split and not any(k in q for k in ['gold','query_gold']) for q in inputs);batches=read(r/'inputs/BATCH_IDS.json');selected=[q['id'] for q in inputs] if a.batch=='all' else batches[a.batch];byid={q['id']:q for q in inputs};assert set(selected)<=set(byid);qs=[byid[key] for key in selected]
 for q in qs:assert q['R']==len(q['ids'])-1 and hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256']
 assert read('/root/autodl-tmp/ouro-e7/reencoding/full_capture/COMPLETE.json')['status']=='COMPLETE' and read('/root/autodl-tmp/ouro-e7/reencoding/full_capture/COMPLETE.json')['n']==22280,'Wait for existing E7 full GPU capture completion'
 transformers.utils.import_utils._torchvision_available=False;from transformers import AutoModelForCausalLM
 lock=open('/root/autodl-tmp/ouro-e6a/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_grad_enabled(False);torch.set_num_threads(1);torch.manual_seed(2021);torch.backends.cuda.matmul.allow_tf32=False;assert torch.__version__=='2.7.1+cu128' and torch.version.cuda=='12.8' and transformers.__version__=='4.55.0' and 'RTX 5090' in torch.cuda.get_device_name();assert sha(Path(a.model)/'model.safetensors')==MODEL_SHA and sha(Path(a.model)/'modeling_ouro.py')==MODEL_SOURCE;assert sha(Path(a.model)/'config.json')==CONFIG_SHA and sha(Path(a.model)/'configuration_ouro.py')==CONFIGURATION_SHA
 identity={'collector_sha256':sha(__file__),'branch_interfaces_sha256':sha(r/'branch_interfaces_v2.py'),'order_sha256':sha(r/'ASTRA_ORDER_C1_UPSTREAM_TO_L18.md'),'cohort_freeze_sha256':sha(r/'inputs/COHORT_AND_OPERATOR_FREEZE.json'),'model_sha256':MODEL_SHA,'model_source_sha256':MODEL_SOURCE,'configuration_source_sha256':sha(Path(a.model)/'configuration_ouro.py'),'config_sha256':sha(Path(a.model)/'config.json'),'inputs_train_sha256':sha(r/'inputs/train.jsonl'),'inputs_validation_sha256':sha(r/'inputs/validation.jsonl'),'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),'dtype':'bfloat16','attention':'sdpa','use_cache':False,'stage_semantics':'Fresh samehardware N/P/U first; exact Q9conditioned per-site reference subtraction; independent postadd x/a composition retaining U actualQKV'};ip=r/'C1_PRODUCER_IDENTITY_v2.json'
 if ip.exists():assert read(ip)==identity
 else:write(ip,identity)
 model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;model.model.total_ut_steps=2;assert sha(inspect.getfile(model.__class__))==MODEL_SOURCE and sha(inspect.getfile(model.config.__class__))==identity['configuration_source_sha256'];versions=[(n,id(v),v._version) for n,v in model.named_parameters()];start=time.perf_counter()
 try:
  with torch.inference_mode():
   if a.stage=='reference':result=references(model,qs,r,identity,a.native_root)
   else:result=treatments(model,qs,r,identity,read(r/'inputs'/(a.split+'_donor_map.json')),[q['id'] for q in inputs],a.technical,batches['independent_replay'])
  assert versions==[(n,id(v),v._version) for n,v in model.named_parameters()];receiptfile=r/(a.split.upper()+'_'+a.stage.upper()+'_'+a.batch.upper()+'_v2_COMPLETE.json');write(receiptfile,{'status':'COMPLETE','stage':a.stage,'split':a.split,'batch':a.batch,'n':len(qs),'source_identity':identity,'source_selected_ids':selected,'wall_seconds_including_no_model_load':time.perf_counter()-start,'model_weights_modified':False,'technical':a.technical,'result':result})
  if a.stage=='treatment' and not a.technical:aggregate_coverage(r,a.split,inputs,identity)
  print('C1_STAGE_COMPLETE',str(receiptfile),flush=True)
 except BaseException as ex:write(r/(a.split.upper()+'_'+a.stage.upper()+'_'+a.batch.upper()+'_v2_FAILED.json'),{'status':'FAILED','exception':repr(ex),'source_identity':identity,'seconds':time.perf_counter()-start});raise
if __name__=='__main__':main()
