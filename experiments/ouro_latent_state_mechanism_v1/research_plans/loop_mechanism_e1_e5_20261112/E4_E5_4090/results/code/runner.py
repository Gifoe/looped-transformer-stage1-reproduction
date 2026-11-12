"""Astra order 001, two-pass native/hybrid collection and controlled forward replay."""
import argparse,os,json,hashlib,time,resource,fcntl
from pathlib import Path
import torch,psutil
import transformers
transformers.utils.import_utils._torchvision_available=False
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file,load_file
from observer import observe,hybrid,bits,thash
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=Path(__file__).resolve().parent
PAIRS=[(1,2),(2,1),(2,3),(3,2),(3,4),(4,3)]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024**2),b''):h.update(x)
 return h.hexdigest()
def write(p,data):
 p=Path(p);temp=p.with_suffix(p.suffix+'.partial');temp.write_text(json.dumps(data,indent=2));os.replace(temp,p)
def key(q):return q['split']+'_'+hashlib.sha256(q['id'].encode()).hexdigest()
def measure():return {'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'cpu_rss_bytes':psutil.Process().memory_info().rss,'cpu_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
def run(model,q,depth,patch=None,capture=False):
 ids=torch.tensor([q['ids']],device='cuda');R=q['R'];assert 0<=R<len(q['ids'])
 assert q['input_ids_sha256']==hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()
 model.model.total_ut_steps=depth;events=[]
 def root(m,args,kw):
  assert bits(args[0],ids) and kw['labels'] is None and kw['use_cache'] is False and kw.get('past_key_values') is None;events.append(1)
 rh=model.register_forward_pre_hook(root,with_kwargs=True)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
 try:
  with observe(model,depth,R,patch,capture) as rec:y=model(ids,labels=None,use_cache=False,exit_at_step=depth-1)
 finally:rh.remove()
 assert events==[1]
 z=y.logits[0,R,:].clone();assert z.dtype==torch.bfloat16 and torch.isfinite(z).all()
 torch.cuda.synchronize();elapsed=time.monotonic()-started
 # Historical production protocol casts BF16 logits to FP32 before log_softmax.
 scores=torch.log_softmax(z.float(),dim=-1);idx=q['targets']
 row={'candidate_logits':[float(z[i]) for i in idx],'scores':[float(scores[i]) for i in idx],'scores_bf16_sensitivity':[float(x) for x in torch.log_softmax(z,dim=-1)[idx]],'logsumexp_fp32':float(torch.logsumexp(z.float(),dim=-1)),'full_logits_sha256':thash(z),'wall_seconds':elapsed,**measure()}
 return row,rec
def collect(model,rows,out,identity):
 for n,q in enumerate(rows):
  p=out/(key(q)+'.native.json');tp=p.with_suffix('.safetensors')
  if p.exists():
   old=json.loads(p.read_text());assert old['identity']==identity and old['tensor_sha256']==sha(tp);continue
  t={};native={};captures={};receipts={};started=time.monotonic()
  for d in (1,2,3,4):
   row,rec=run(model,q,d,capture=True);native[str(d)]=row;captures[d]=rec.pop('qkv');receipts[str(d)]={k:v for k,v in rec.items() if not torch.is_tensor(v)}
   for name in ('O','a_R','x_R','final_norm_R'):t[f'K{d}_{name}']=rec[name].cpu().contiguous()
  for receiver,donor in PAIRS:
   patch=hybrid(captures[receiver],captures[donor],q['R']);t[f'{receiver}_{donor}_target']=patch.cpu().contiguous()
   p32=hybrid(captures[receiver],captures[donor],q['R'],True);t[f'{receiver}_{donor}_fp32_sdpa_bf16_writeback']=p32.cpu().contiguous()
  del captures
  temp=Path(str(tp)+'.partial');save_file(t,str(temp));back=load_file(str(temp));assert all(bits(v,back[k]) for k,v in t.items());os.replace(temp,tp)
  write(p,{'id':q['id'],'split':q['split'],'input_ids_sha256':hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest(),'metadata_sha256':q['metadata_sha256'],'R':q['R'],'targets':q['targets'],'identity':identity,'native':native,'capture_receipts':receipts,'tensor_sha256':sha(tp),'total_wall_seconds':time.monotonic()-started})
  print('NATIVE_COMPLETE',n+1,len(rows),q['split'],q['id'],time.monotonic()-started,flush=True)
def replay(model,rows,out,identity,precision_ids):
 # Frozen cyclic donors are based on ID sort and independent of outcomes.
 ordered=rows;assert len(ordered)>=2
 donor_map={q['id']:ordered[(i+1)%len(ordered)] for i,q in enumerate(ordered)}
 write(out/'control_donor_map.json',{k:v['id'] for k,v in donor_map.items()})
 for n,q in enumerate(rows):
  p=out/(key(q)+'.results.json');nativep=out/(key(q)+'.native.json');m=json.loads(nativep.read_text());t=load_file(str(nativep.with_suffix('.safetensors')))
  if p.exists():
   prev=json.loads(p.read_text());assert prev['identity']==identity;continue
  result=[];saved={};started=time.monotonic()
  for receiver,donor in PAIRS:
   base=t[f'K{receiver}_O'];target=t[f'{receiver}_{donor}_target'];delta=target.float()-base.float();oid=donor_map[q['id']];fallback=[]
   for unused in range(len(ordered)-1):
    ot=load_file(str((out/(key(oid)+'.native.json')).with_suffix('.safetensors')))
    other=ot[f'{receiver}_{donor}_target'].float()-ot[f'K{receiver}_O'].float()
    if float(other.norm())>0:break
    fallback.append(oid['id']);oid=donor_map[oid['id']]
   assert oid['id']!=q['id']
   assert float(base.float().norm())>0 and float(other.norm())>0
   norm=(base.float()*(target.float().norm()/base.float().norm())).to(base.dtype)
   unrelated=(base.float()+delta.norm()*other/other.norm()).to(base.dtype)
   controls={'target':target,'norm_only':norm,'unrelated_delta':unrelated}
   # Identity is direction-invariant; one exact reinjection per recipient.
   if receiver not in saved:
    row,rec=run(model,q,receiver,base);assert row['full_logits_sha256']==m['native'][str(receiver)]['full_logits_sha256'];saved[receiver]=row
    result.append({'receiver_loop':receiver,'donor_loop':receiver,'condition':'identity','donor_id':q['id'],**row,'outside_R_unchanged':rec['outside_R_unchanged']})
   if q['id'] in precision_ids:controls['fp32_sdpa_bf16_writeback']=t[f'{receiver}_{donor}_fp32_sdpa_bf16_writeback']
   for condition,patch in controls.items():
    row,rec=run(model,q,receiver,patch);assert bits(rec['O'].cpu(),base)
    pr=patch.float();bf=base.float()
    result.append({'pair':f'{min(receiver,donor)}-{max(receiver,donor)}','direction':'rollback' if receiver>donor else 'restore','receiver_loop':receiver,'donor_loop':donor,'condition':condition,'donor_id':oid['id'] if condition=='unrelated_delta' else q['id'],'donor_fallbacks':fallback if condition=='unrelated_delta' else [],'native':m['native'][str(receiver)],'native_sdpa_R_sha256':thash(base),'injected_sha256':thash(patch),'a_R_sha256':thash(rec['a_R']),'final_norm_R_sha256':thash(rec['final_norm_R']),'target_delta_norm':float(delta.norm()),'actual_delta_norm':float((pr-bf).norm()),'native_norm':float(bf.norm()),'injected_norm':float(pr.norm()),'cosine':float(torch.nn.functional.cosine_similarity(pr.flatten(),bf.flatten(),dim=0)),'outside_R_unchanged':rec['outside_R_unchanged'],**row})
    print('CONDITION_COMPLETE',q['id'],receiver,donor,condition,round(row['wall_seconds'],3),flush=True)
  write(p,{'id':q['id'],'split':q['split'],'identity':identity,'native_file_sha256':sha(nativep),'rows':result,'total_wall_seconds':time.monotonic()-started})
  print('QUESTION_COMPLETE',n+1,len(rows),q['split'],q['id'],time.monotonic()-started,flush=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('--inputs',required=True);a.add_argument('--cohort',required=True);a.add_argument('--output',required=True);a.add_argument('--phase',choices=['collect','replay','all'],default='all');a.add_argument('--limit',type=int);args=a.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True);lock=open(ROOT/'gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cohort=json.loads(Path(args.cohort).read_text());selected=cohort['ids'];precision_ids=set(cohort.get('precision_ids',[]))
 if not precision_ids:
  groups={}
  for question in cohort.get('questions',[]):groups.setdefault(question['historical_group'],[]).append(question['id'])
  precision_ids={i for group in groups.values() for i in group[:2]}
 allrows=[json.loads(x) for x in Path(args.inputs).read_text().splitlines() if x.strip()];byid={q['id']:q for q in allrows};rows=[byid[i] for i in selected];assert len(rows)==len(set(selected))
 if args.limit:rows=rows[:args.limit]
 assert len(rows)>=2 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 modeldir=ROOT/'model';assert sha(modeldir/'model.safetensors')=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
 identity={'model_sha256':sha(modeldir/'model.safetensors'),'model_source_sha256':sha(modeldir/'modeling_ouro.py'),'config_sha256':sha(modeldir/'config.json'),'runner_sha256':sha(__file__),'observer_sha256':sha(ROOT/'observer.py'),'inputs_sha256':sha(args.inputs),'cohort_sha256':sha(args.cohort),'torch':torch.__version__,'transformers':__import__('transformers').__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'attention':'sdpa','dtype':'bfloat16','use_cache':False,'batch_size':1}
 identity['actual_selected_ids_sha256']=hashlib.sha256(json.dumps([q['id'] for q in rows]).encode()).hexdigest();identity['actual_precision_ids']=sorted(precision_ids.intersection(q['id'] for q in rows))
 previous=out/'execution_identity.json'
 if previous.exists():assert json.loads(previous.read_text())==identity
 write(previous,identity);started=time.monotonic()
 model=AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False
 versions=[(n,id(p),p._version) for n,p in model.named_parameters()]
 with torch.inference_mode():
  if args.phase in ('collect','all'):collect(model,rows,out,identity)
  if args.phase in ('replay','all'):replay(model,rows,out,identity,precision_ids)
 assert versions==[(n,id(p),p._version) for n,p in model.named_parameters()]
 write(out/'receipt.json',{'status':'COMPLETE','N':len(rows),'phase':args.phase,'identity':identity,'wall_seconds':time.monotonic()-started,**measure()})
if __name__=='__main__':main()
