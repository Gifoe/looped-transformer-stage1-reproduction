"""Train-only eight-corner production collector; never starts validation/test."""
import os,json,hashlib,fcntl,time,uuid,resource,shutil
from pathlib import Path
import torch,psutil
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file,load_file
import eight_observer,eight_runner,replay_eight,recovery_adapter,qkv_runner,natural_l18_hook,natural_sources,technical,full_input,state_capture,source_input,probe_eight
from source_input import ROOT,sha,initial_sources
from state_capture import bits
from probe_eight import model_hash,write_json,require,thash
if not __debug__:raise RuntimeError('optimized Python forbidden')
BASE=ROOT/'research_plans/natural_qkv_eight_corner_20261112';OUT=BASE/'train';N={f'G{i}{j}{k}' for i in (1,2) for j in (1,2) for k in (1,2)}
def check_existing(ident,sources,codes):
 key=hashlib.sha256(ident.encode()).hexdigest();mp=OUT/(key+'.json');tp=mp.with_suffix('.safetensors')
 if not mp.exists() and not tp.exists():return False
 require(mp.exists() and tp.exists(),'orphan pair');m=json.loads(mp.read_text());src,old,op=recovery_adapter.load('train',ident);e=src['natural'];ot=load_file(str(op));t=load_file(str(tp));require(m['id']==ident and m['split']=='train' and m['tensor_sha256']==sha(tp) and m['source_hashes']==sources and m['code_hashes']==codes and m['targets']==e['targets'] and m['gold']==e['gold'] and m['R']==e['R'] and m['input_ids_sha256']==hashlib.sha256(json.dumps(e['ids']).encode()).hexdigest() and m['natural_metadata_sha256']==e['metadata_sha256'] and m['natural_tensor_sha256']==e['tensor_sha256'] and m['actual_branch_metadata_sha256']==src['source_metadata_sha256'] and m['actual_branch_tensor_sha256']==src['source_tensor_sha256'] and m['old_qkv_metadata_sha256']==sha(op.with_suffix('.json')) and m['old_qkv_tensor_sha256']==sha(op),'source identity');require(set(t)=={n+'_'+v for n in N for v in ('logits','sdpa_R','a_R')} and set(m['scores'])==N,'keys')
 for n in N:
  cr=m['corner_receipts'][n];shape=cr['actual_receiver_qkv']['qkv']['q']['shape'];z=t[n+'_logits'];sd=t[n+'_sdpa_R'];a=t[n+'_a_R'];require(z.dtype==torch.bfloat16 and tuple(z.shape)==(1,49152) and sd.dtype==torch.bfloat16 and list(sd.shape)==[shape[0],shape[1],shape[3]] and a.dtype==torch.bfloat16 and tuple(a.shape)==(1,2048) and all(torch.isfinite(x).all() for x in (z,sd,a)) and thash(sd)==cr['injected_sdpa_R_sha256'] and thash(a)==cr['a_R_sha256'] and cr['outside_R_unchanged'] is True and cr['root_input_events']==1 and technical.score(z.to('cuda'),e['targets'])==m['scores'][n],'saved replay')
 for n,o in [('G111','F11'),('G211','F21'),('G122','F12'),('G222','F22')]:require(all(bits(t[n+'_'+v],ot[o+'_'+v]) for v in ('logits','sdpa_R','a_R')) and m['scores'][n]==old['scores'][o],'old four corner')
 return True
def main():
 # This entrypoint is deliberately train-only; later splits require new explicit gate.
 review=BASE/'ROUND1_ASTRA_PROBE_REVIEW.json';probe=BASE/'probe';rv=json.loads(review.read_text());require(rv['status']=='ASTRA_APPROVED_THREE_CASE_PROBE' and rv['receipt_sha256']==sha(probe/'receipt.json'),'Astra review');receipt=json.loads((probe/'receipt.json').read_text());replay=json.loads((probe/'replay.json').read_text());require(receipt['pending_sha256']==sha(probe/'probe_pending.json') and receipt['replay_sha256']==sha(probe/'replay.json') and replay['pending_sha256']==sha(probe/'probe_pending.json') and replay['status']=='EIGHT_CORNER_THREE_CASE_CUDA_REPLAY_PASSED','probe evidence chain');pending=json.loads((probe/'probe_pending.json').read_text());recovery_adapter.initialize();lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=True);require(not (OUT/'receipt.json').exists(),'already complete');paths={**initial_sources(),'terms':BASE/'ROUND1_PROTOCOL.md','l18_coverage':qkv_runner.LCOV,'recovery_zip':recovery_adapter.ZIP,'old_qkv_coverage':recovery_adapter.QGPU,'probe_receipt':probe/'receipt.json','probe_pending':probe/'probe_pending.json','review':review};sources={k:sha(v) for k,v in paths.items()};mods=(eight_observer,eight_runner,replay_eight,recovery_adapter,qkv_runner,natural_l18_hook,natural_sources,technical,full_input,state_capture,source_input);cp={'train':Path(__file__).resolve(),'probe':Path(probe_eight.__file__).resolve(),**{m.__name__:Path(m.__file__).resolve() for m in mods}};codes={k:sha(v) for k,v in cp.items()};require(codes['probe']==pending['source_hashes']['probe'] and all(codes[m.__name__]==pending['source_hashes']['code_'+m.__name__] for m in mods),'probe source drift');stats={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in list(paths.values())+list(cp.values())};require(not list(OUT.glob('*.partial')),'partial requires review');require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA');torch.cuda.reset_peak_memory_stats()
  with torch.no_grad():model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  model.config.use_cache=False;model.model.config.use_cache=False;mh=model_hash(model);require(mh==pending['model_state_sha256'],'model drift');params=lambda:[(n,id(t),t.data_ptr(),t._version,str(t.dtype),str(t.device),tuple(t.shape)) for n,t in list(model.named_parameters())+list(model.named_buffers())];versions=params()
  def guard():require(params()==versions and all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in stats.items()),'model/source drift')
  rows=full_input.coverage('train');require(len(rows)==1119 and len({q['id'] for q in rows})==1119,'cohort');done=0;started=time.monotonic()
  for q in rows:
   guard();require(shutil.disk_usage(ROOT).free>10*1024**3 and psutil.virtual_memory().available>4*1024**3,'disk/RAM');ident=q['id'];key=hashlib.sha256(ident.encode()).hexdigest()
   if check_existing(ident,sources,codes):done+=1;continue
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.monotonic();r=eight_runner.question(model,'train',ident,guard);torch.cuda.synchronize();elapsed=time.monotonic()-t0;allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();logits=r.pop('logits');inj=r.pop('injections');a=r.pop('a_states');require(set(logits)==set(inj)==set(a)==N,'conditions');t={n+'_'+suffix:v.detach().cpu().contiguous().clone() for suffix,dd in [('logits',logits),('sdpa_R',inj),('a_R',a)] for n,v in dd.items()};tp=OUT/(key+'.safetensors');tmp=OUT/(key+'.'+uuid.uuid4().hex+'.partial')
   try:
    save_file(t,str(tmp));back=load_file(str(tmp));require(set(t)==set(back) and all(bits(t[k],back[k]) for k in t),'readback');os.link(tmp,tp)
   finally:
    if tmp.exists():tmp.unlink()
   scores=r['scores'];r['predictions']={k:max(range(len(v)),key=lambda j:v[j]) for k,v in scores.items()};r['top_ties']={k:sum(x==max(v) for x in v)>1 for k,v in scores.items()};r['target_logits']={n:[float(t[n+'_logits'][0,j]) for j in r['targets']] for n in N};r['runtime_measurement']={'question_wall_seconds_cuda_synchronized':elapsed,'peak_allocated':allocated,'peak_reserved':reserved};r['tensor_sha256']=sha(tp);r['source_hashes']=sources;r['code_hashes']=codes;write_json(OUT/(key+'.json'),r);require(check_existing(ident,sources,codes),'saved verification');done+=1;print('TRAIN_QUESTION_COMPLETE',done,ident,flush=True)
  guard();require(done==1119 and {k:sha(v) for k,v in paths.items()}==sources and {k:sha(v) for k,v in cp.items()}==codes and model_hash(model)==mh,'final');write_json(OUT/'receipt.json',{'status':'TRAIN_COMPLETE_PENDING_INDEPENDENT_COVERAGE','split':'train','count':done,'sources':sources,'codes':codes,'model_state_sha256':mh,'runtime':pending['runtime'],'elapsed_s':time.monotonic()-started,'rss_peak_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss});print('TRAIN_COMPLETE_PENDING_INDEPENDENT_COVERAGE',flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
