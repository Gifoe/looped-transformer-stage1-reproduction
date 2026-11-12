"""Capture pre-frozen 64 train discovery cases; no scientific effect summaries."""
import os,sys,json,hashlib,time,fcntl,uuid,shutil,torch
from pathlib import Path
from safetensors.torch import load_file,save_file
from transformers import AutoModelForCausalLM
import recovery_adapter,probe_eight,technical,e1_question,e1_six_question,e1_control_core,e1_injection_observer,e1_donor_pool,e1_fp64_attention,eight_observer,eight_runner,qkv_runner,natural_l18_hook,natural_sources,full_input,source_input,state_capture,replay_eight
ROOT=Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E1';OUT=E/'train';P=E/'technical_probe';S=E/'six_corner_probe';SELECTION=B/'orders/E1_DISCOVERY64_FROZEN.json';N=['G111','G112','G121','G211','G212','G221']
if not __debug__:raise RuntimeError('optimized Python forbidden')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def need(x,s):
 if not x:raise ValueError(s)
def replay_existing(ident,sources):
 key=hashlib.sha256(ident.encode()).hexdigest();mp=OUT/(key+'.json');tp=mp.with_suffix('.safetensors')
 if not mp.exists() and not tp.exists():return False
 need(mp.exists() and tp.exists(),'orphan requires review');m=json.loads(mp.read_text());need(m['id']==ident and m['split']=='train' and m['source_hashes']==sources and m['tensor_sha256']==sha(tp),'resume identity');t=load_file(str(tp));old,ot,_=e1_question.read_old('train',ident);need(m['targets']==old['targets'] and m['R']==old['R'] and m['input_ids_sha256']==old['input_ids_sha256'],'resume input identity');scores=m['controls']['conditions']
 for q in (1,2):
  for name in ('ID','ADD','ADD_NORM','UNRELATED'):
   if name=='ADD_NORM' and scores[str(q)]['ADD_NORM_status']=='UNDEFINED':continue
   z=t[f'q{q}_{name}_logits'];need(z.dtype==torch.bfloat16 and tuple(z.shape)==(1,49152) and torch.isfinite(z).all() and technical.score(z.cuda(),m['targets'])==scores[str(q)][name]['scores'],'resume score')
   if name=='ID':need(torch.equal(z,ot[f'G{q}22_logits']) and torch.equal(t[f'q{q}_{name}_attention_localnorm'],ot[f'G{q}22_a_R']),'resume ID')
  need(torch.equal(t[f'q{q}_restore'],ot[f'G{q}22_sdpa_R']),'resume restore tensor')
 for n in N:need(torch.equal(t['old_'+n+'_logits'],ot[n+'_logits']) and torch.equal(t['old_'+n+'_attention_localnorm'],ot[n+'_a_R']) and torch.equal(t['old_'+n+'_injected_sdpa_R'],ot[n+'_sdpa_R']) and technical.score(t['old_'+n+'_logits'].cuda(),m['targets'])==m['old_stages']['conditions'][n]['scores'],'resume six')
 return True
def main():
 selection=json.loads(SELECTION.read_text());need(selection['status']=='PRE_EFFECT_DISCOVERY64_FROZEN' and selection['count']==64 and sha(SELECTION)=='6faad196db958744780ee6477af8839c693798b8b24d2a01a2d450c856c720e1','selection');ids=[v['id'] for group in ('rescue','damage','stable_correct','stable_wrong') for v in selection['groups'][group]];need(len(ids)==len(set(ids))==64,'IDs')
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=True);need(not (OUT/'discovery64_receipt.json').exists(),'discovery already complete');need(not list(OUT.glob('*.partial')),'partial requires manual review');p=json.loads((P/'pending.json').read_text());pr=json.loads((P/'replay.json').read_text());sp=json.loads((S/'pending.json').read_text());sr=json.loads((S/'replay.json').read_text());need(pr['status']=='E1_THREE_CASE_CUDA_SCORE_AND_ENDPOINT_REPLAY_PASSED' and pr['pending_sha256']==sha(P/'pending.json') and sr['status']=='E1_SIX_CORNER_CUDA_SCORE_AND_OLD_ANCHOR_REPLAY_PASSED' and sr['pending_sha256']==sha(S/'pending.json') and all(sha(Path(k))==v for k,v in p['source_hashes'].items()) and all(sha(Path(k))==v for k,v in sp['source_hashes'].items()),'technical proof');
  modules=(e1_question,e1_six_question,e1_control_core,e1_injection_observer,e1_donor_pool,e1_fp64_attention,eight_observer,eight_runner,qkv_runner,natural_l18_hook,natural_sources,full_input,source_input,state_capture,replay_eight,recovery_adapter,probe_eight,technical);files=[Path(__file__).resolve(),SELECTION,P/'pending.json',P/'replay.json',S/'pending.json',S/'replay.json',B/'orders/ORDER_E1_ADD_CONTROLS_v1.md']+[Path(m.__file__).resolve() for m in modules];sources={str(x):sha(x) for x in files};recovery_adapter.initialize();pool=e1_donor_pool.Pool();need(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA');need(torch.__version__==p['runtime']['torch'] and torch.version.cuda==p['runtime']['cuda'] and torch.cuda.get_device_name()==p['runtime']['gpu'],'runtime');start=time.monotonic()
  with torch.no_grad():model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  model.config.use_cache=False;model.model.config.use_cache=False;mh=probe_eight.model_hash(model);need(mh==p['model_state_sha256'],'model');params=lambda:[(n,id(x),x.data_ptr(),x._version) for n,x in list(model.named_parameters())+list(model.named_buffers())];version=params()
  def guard():need(params()==version,'model mutation')
  done=0
  for ident in ids:
   guard();need(shutil.disk_usage(ROOT).free>5*1024**3,'disk space');key=hashlib.sha256(ident.encode()).hexdigest()
   if replay_existing(ident,sources):done+=1;continue
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.monotonic();cm,ct=e1_question.run_case(model,ident,pool,guard,'train',include_restore=False,save_kv=False);sm,st=e1_six_question.one(model,ident,guard,'train');torch.cuda.synchronize();wall=time.monotonic()-t0;need(cm['id']==sm['id']==ident and cm['R']==sm['R'] and cm['input_ids_sha256']==sm['input_ids_sha256'] and cm['targets']==sm['targets'],'paired source');t={**ct,**{'old_'+k:v for k,v in st.items()}};need(len(t)==len(ct)+len(st),'key collision');tp=OUT/(key+'.safetensors');tmp=OUT/(key+'.'+uuid.uuid4().hex+'.partial')
   try:
    save_file(t,str(tmp));back=load_file(str(tmp));need(set(back)==set(t) and all(torch.equal(back[k],t[k]) for k in t),'tensor readback');os.link(tmp,tp)
   finally:
    if tmp.exists():tmp.unlink()
   m={'id':ident,'split':'train','R':cm['R'],'targets':cm['targets'],'input_ids_sha256':cm['input_ids_sha256'],'source_hashes':sources,'controls':cm,'old_stages':sm,'tensor_sha256':sha(tp),'tensor_bytes':tp.stat().st_size,'question_wall_seconds_cuda_synchronized_excludes_save':wall,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'disk_free_after_save':shutil.disk_usage(ROOT).free};probe_eight.write_json(OUT/(key+'.json'),m);need(replay_existing(ident,sources),'written shard verification');done+=1;print('E1_DISCOVERY_QUESTION',done,ident,flush=True)
  need(done==64 and all(sha(Path(k))==v for k,v in sources.items()) and probe_eight.model_hash(model)==mh,'final integrity');probe_eight.write_json(OUT/'discovery64_receipt.json',{'status':'DISCOVERY64_COLLECTED_PENDING_INDEPENDENT_CUDA_AUDIT','count':done,'selection_sha256':sha(SELECTION),'sources':sources,'model_state_sha256':mh,'runtime':p['runtime'],'elapsed_seconds':time.monotonic()-start,'raw_kv_scope':'full K/V transient, not archived; capture descriptor/hash recorded; technical cases retain raw','questions':[{'id':ident,'metadata_sha256':sha(OUT/(hashlib.sha256(ident.encode()).hexdigest()+'.json')),'tensor_sha256':sha(OUT/(hashlib.sha256(ident.encode()).hexdigest()+'.safetensors'))} for ident in ids]});print('E1_DISCOVERY64_PENDING_CUDA_AUDIT',flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
