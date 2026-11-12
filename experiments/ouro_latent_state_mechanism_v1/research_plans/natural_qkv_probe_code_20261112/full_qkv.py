"""Full QKV factorial collection; requires independent three-case approval."""
import os,json,hashlib,fcntl,time,uuid,resource,shutil
from pathlib import Path
import torch,psutil
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file,load_file
import qkv_observer,qkv_runner,replay_qkv,natural_l18_hook,natural_sources,technical,full_input,state_capture,source_input
from source_input import ROOT,initial_sources,sha
from state_capture import bits
from probe_qkv import model_hash,write_json,require,thash
OUT=ROOT/'research_plans/natural_qkv_full_20261112'
TERMS=ROOT/'research_plans/NATURAL_L18_QR_KV_FACTORIAL_EXECUTION_TERMS_20261112.md'
REVIEW=ROOT/'research_plans/NATURAL_QKV_ASTRA_PROBE_REVIEW_20261112.json'
NAMES={'F11','F21','F12','F22'}
def check(split,ident,sources,codes):
 key=split+'_'+hashlib.sha256(ident.encode()).hexdigest();mp=OUT/(key+'.json');tp=OUT/(key+'.safetensors')
 if not mp.exists() and not tp.exists():return False
 require(mp.exists() and tp.exists(),'orphan pair');m=json.loads(mp.read_text());t=load_file(str(tp));src=natural_sources.load(split,ident);e=src['natural']
 require(m['id']==ident and m['split']==split and m['tensor_sha256']==sha(tp) and m['source_hashes']==sources and m['code_hashes']==codes,'resume binding')
 require(m['targets']==e['targets'] and m['gold']==e['gold'] and m['R']==e['R'] and m['input_ids_sha256']==hashlib.sha256(json.dumps(e['ids']).encode()).hexdigest() and m['natural_metadata_sha256']==e['metadata_sha256'] and m['natural_tensor_sha256']==e['tensor_sha256'] and m['branch_metadata_sha256']==src['source_metadata_sha256'] and m['branch_tensor_sha256']==src['source_tensor_sha256'],'source')
 require(set(m['scores'])==NAMES and set(t)=={f'{n}_{s}' for n in NAMES for s in ('logits','sdpa_R','a_R')},'keys')
 for n in NAMES:
  cr=m['corner_receipts'][n];q=cr['actual_receiver_qkv']['qkv']['q']['shape'];z=t[n+'_logits'];sd=t[n+'_sdpa_R'];a=t[n+'_a_R']
  require(len(q)==4 and sd.dtype==torch.bfloat16 and list(sd.shape)==[q[0],q[1],q[3]] and a.dtype==torch.bfloat16 and a.shape==(1,2048) and z.dtype==torch.bfloat16 and z.shape==(1,49152) and all(torch.isfinite(x).all() for x in (z,sd,a)) and thash(sd)==cr['injected_sdpa_R_sha256'] and thash(a)==cr['a_R_sha256'] and cr['outside_R_unchanged'] is True and cr['root_input_events']==1 and technical.score(z.to('cuda'),e['targets'])==m['scores'][n],'replay')
 lp=ROOT/'research_plans/natural_l18_full_run_20261111';require(m['l18_coverage_sha256']==sha(qkv_runner.LCOV) and m['l18_metadata_sha256']==sha(lp/(key+'.json')) and m['l18_tensor_sha256']==sha(lp/(key+'.safetensors')),'L18 source');l18=load_file(str(lp/(key+'.safetensors')))
 require(bits(t['F11_logits'],e['logits'][0]) and bits(t['F22_logits'],l18['T12']) and bits(t['F11_a_R'],src['states'][0]['a']) and bits(t['F22_a_R'],src['states'][1]['a']),'endpoint')
 return True
def main():
 require(sha(TERMS)=='c42fa7dd4021e4a77fcf68e9da110024f3d8df44e810aaa4d40ae21fea534b61','terms');rev=json.loads(REVIEW.read_text());require(rev['status']=='QKV_PROBE_APPROVED' and rev['receipt_sha256']==sha(ROOT/'research_plans/natural_qkv_probe_20261112/receipt.json'),'review')
 pr=ROOT/'research_plans/natural_qkv_probe_20261112';receipt=json.loads((pr/'receipt.json').read_text());require(receipt['pending_sha256']==sha(pr/'probe_pending.json'),'pending receipt');pending=json.loads((pr/'probe_pending.json').read_text())
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=True);require(not (OUT/'receipt.json').exists(),'complete')
  paths={**initial_sources(),'terms':TERMS,'review':REVIEW,'l18_coverage':qkv_runner.LCOV,'probe_receipt':pr/'receipt.json','probe_pending':pr/'probe_pending.json'};sources={k:sha(v) for k,v in paths.items()};mods=(qkv_observer,qkv_runner,replay_qkv,natural_l18_hook,natural_sources,technical,full_input,state_capture,source_input);cp={'full':Path(__file__).resolve(),'probe':Path(__import__('probe_qkv').__file__).resolve(),**{m.__name__:Path(m.__file__).resolve() for m in mods}};codes={k:sha(v) for k,v in cp.items()};require(all(codes[m.__name__]==pending['source_hashes']['code_'+m.__name__] for m in mods),'probe core drift');stats={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in list(paths.values())+list(cp.values())}
  require(not list(OUT.glob('*.partial')),'partial requires inspection');require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'cuda');torch.cuda.reset_peak_memory_stats()
  with torch.no_grad():model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  model.config.use_cache=False;model.model.config.use_cache=False;mh=model_hash(model);require(mh==pending['model_state_sha256'],'probe model drift');params=lambda:[(n,id(t),t.data_ptr(),t._version,str(t.dtype),str(t.device),tuple(t.shape)) for n,t in list(model.named_parameters())+list(model.named_buffers())];vers=params()
  def guard():require(params()==vers and all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in stats.items()),'drift')
  counts={s:0 for s in ('train','validation','test')};start=time.monotonic()
  for split in counts:
   for row in full_input.coverage(split):
    guard();require(shutil.disk_usage(ROOT).free>20*1024**3 and psutil.virtual_memory().available>4*1024**3,'disk/RAM');ident=row['id'];key=split+'_'+hashlib.sha256(ident.encode()).hexdigest()
    if check(split,ident,sources,codes):counts[split]+=1;continue
    r=qkv_runner.question(model,split,ident,guard);logits=r.pop('logits');inj=r.pop('injections');a=r.pop('a_states');require(set(logits)==set(inj)==set(a)==NAMES,'conditions');t={f'{n}_{suffix}':v.detach().cpu().contiguous().clone() for suffix,d in [('logits',logits),('sdpa_R',inj),('a_R',a)] for n,v in d.items()};tp=OUT/(key+'.safetensors');tmp=OUT/(key+'.'+uuid.uuid4().hex+'.partial')
    try:
     save_file(t,str(tmp));back=load_file(str(tmp));require(set(t)==set(back) and all(bits(t[k],back[k]) for k in t),'readback');os.link(tmp,tp)
    finally:
     if tmp.exists():tmp.unlink()
    scores=r['scores'];r['predictions']={k:max(range(len(v)),key=lambda j:v[j]) for k,v in scores.items()};r['top_ties']={k:sum(x==max(v) for x in v)>1 for k,v in scores.items()};r['tensor_sha256']=sha(tp);r['source_hashes']=sources;r['code_hashes']=codes;write_json(OUT/(key+'.json'),r);require(check(split,ident,sources,codes),'saved verification');counts[split]+=1;print('QUESTION_COMPLETE',split,counts[split],ident,flush=True)
  guard();require(counts=={'train':1119,'validation':299,'test':1172} and {k:sha(v) for k,v in paths.items()}==sources and {k:sha(v) for k,v in cp.items()}==codes and model_hash(model)==mh,'final');write_json(OUT/'receipt.json',{'status':'FULL_QKV_PENDING_INDEPENDENT_COVERAGE','counts':counts,'source_hashes':sources,'code_hashes':codes,'model_state_sha256':mh,'elapsed_s':time.monotonic()-start,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss});print('FULL_QKV_COMPLETE',flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
