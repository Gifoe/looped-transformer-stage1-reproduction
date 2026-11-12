"""Capture the frozen 64 train E3 source rollback cases; no partial effect analysis."""
import os,json,gzip,hashlib,time,fcntl,uuid,torch
from pathlib import Path
from safetensors.torch import load_file,save_file
from transformers import AutoModelForCausalLM
import recovery_adapter,probe_eight,e1_question,e1_injection_observer,e3_source_control_v2,e3_source_question,eight_observer,eight_runner,qkv_runner,natural_l18_hook,natural_sources,technical,full_input,source_input,state_capture,replay_eight
ROOT=Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E3';OUT=E/'source_train64';P=E/'source_technical';A=E/'discovery64/independent_audit.json';SEL=B/'orders/E1_DISCOVERY64_FROZEN.json';LED=ROOT/'research_plans/prompt_boundary_audit_20261111/boundary_ledger.jsonl';COV=ROOT/'research_plans/prefix_question_partition_draft_20261111/saved_offset_coverage.json';REF=ROOT/'research_plans/natural_prefix_identity_20261111/reference_prefix_ids.json';CAT=B/'results/compact_input_catalog.jsonl.gz';ORDER=B/'orders/ORDER_E3_SOURCE_INTERVENTION_v1.md'
if not __debug__:raise RuntimeError('optimized Python forbidden')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def need(x,s):
 if not x:raise ValueError(s)
def thash(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def validate(ident,sources):
 key=hashlib.sha256(ident.encode()).hexdigest();mp=OUT/(key+'.json');tp=mp.with_suffix('.safetensors')
 if not mp.exists() and not tp.exists():return False
 need(mp.exists() and tp.exists(),'orphan');m=json.loads(mp.read_text());need(m['id']==ident and m['split']=='train' and m['source_hashes']==sources and m['tensor_sha256']==sha(tp),'resume identity');old,ot,_=e1_question.read_old('train',ident);need(m['targets']==old['targets'] and m['R']==old['R'] and m['input_ids_sha256']==old['input_ids_sha256'] and m['gold']==old['gold'] and m['fixed_wrong']==old['fixed_wrong'],'source identity');t=load_file(str(tp));names={'ID','K_Q','V_Q','KV_Q','KV_P','K_Q_NORM','V_Q_NORM','V_Q_ROLL'};need(set(m['conditions'])=={'1','2'} and all(set(m['conditions'][str(q)])==names for q in (1,2)),'condition keys');expected={f'q{q}_{name}_{suffix}' for q in (1,2) for name in names if m['conditions'][str(q)][name]['control']['status']=='OK' for suffix in ('logits','sdpa_R','a_R')};need(set(t)==expected,'tensor keys')
 for q in (1,2):
  for name in ('ID','K_Q','V_Q','KV_Q','KV_P','K_Q_NORM','V_Q_NORM','V_Q_ROLL'):
   cm=m['conditions'][str(q)][name]
   if cm['control']['status']!='OK':need(name in ('K_Q_NORM','V_Q_NORM') and cm['control']['status']=='UNDEFINED_ZERO_DONOR_NONZERO_TARGET','bad undefined');continue
   tag=f'q{q}_{name}';z=t[tag+'_logits'];sr=t[tag+'_sdpa_R'];ar=t[tag+'_a_R'];need(z.dtype==sr.dtype==ar.dtype==torch.bfloat16 and tuple(z.shape)==(1,49152) and tuple(sr.shape)==(1,16,128) and tuple(ar.shape)==(1,2048) and all(torch.isfinite(v).all() for v in (z,sr,ar)) and thash(z)==cm['logits_sha256'] and thash(sr)==cm['sdpa_R_sha256'] and thash(ar)==cm['a_R_sha256'] and technical.score(z.cuda(),m['targets'])==cm['scores'] and cm['outside_R_unchanged'],'score')
   if name=='ID':need(torch.equal(z,ot[f'G{q}22_logits']) and torch.equal(t[tag+'_sdpa_R'],ot[f'G{q}22_sdpa_R']) and torch.equal(t[tag+'_a_R'],ot[f'G{q}22_a_R']),'ID endpoint')
 return True
def main():
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);sel=json.loads(SEL.read_text());ids=[v['id'] for g in ('rescue','damage','stable_correct','stable_wrong') for v in sel['groups'][g]];a=json.loads(A.read_text());p=json.loads((P/'pending.json').read_text());r=json.loads((P/'replay.json').read_text());need(len(ids)==len(set(ids))==64 and sha(SEL)=='6faad196db958744780ee6477af8839c693798b8b24d2a01a2d450c856c720e1' and a['status']=='E3_DISCOVERY64_INDEPENDENT_SAVED_PARTITION_AUDIT_PASSED' and a['selection_sha256']==sha(SEL) and sha(A)=='f0b138195e2b88a0b8bf25ad892653a3f5d489c8b561a4b9590eec700d62cd1b' and r['status']=='E3_SOURCE_THREE_CASE_CUDA_REPLAY_PASSED' and r['pending_sha256']==sha(P/'pending.json') and all(sha(Path(k))==v for k,v in p['source_hashes'].items()),'prior evidence');need(sha(ORDER)=='07f9e62111beda94f0b856c96967862ef7bcb24f09a80d043c38eac806b89a94' and sha(LED)=='a77258ade36415efb17ca09a886417f9b265ce53feec1589431dff6062f4e2bc' and sha(COV)=='5263cc8bf07900822b11954194b7e9fd60c3066c9f43d96baee092157445dba6' and sha(REF)=='e984a118a098923ba87fab4ddd4a26d1f9da8c5dc9767c5ab073352cd3dfbf67' and sha(CAT)=='a38218bfe0c9940f326663b8ecc0447f875a18c07f0091813ddfef4a3d12342d','source files');
  with LED.open() as f:ledger={x['id']:x for line in f if (x:=json.loads(line))['split']=='train' and x['id'] in ids}
  with gzip.open(CAT,'rt') as f:cat={x['id']:x for line in f if (x:=json.loads(line))['split']=='train' and x['id'] in ids}
  prefix=json.loads(REF.read_text())['input_ids'];need(len(prefix)==1839 and set(ledger)==set(cat)==set(ids) and all(v['input_ids'][:1839]==prefix for v in cat.values()),'boundary');mods=(recovery_adapter,probe_eight,e1_question,e1_injection_observer,e3_source_control_v2,e3_source_question,eight_observer,eight_runner,qkv_runner,natural_l18_hook,natural_sources,technical,full_input,source_input,state_capture,replay_eight);files=[Path(__file__).resolve(),SEL,A,ORDER,LED,COV,REF,CAT,P/'pending.json',P/'replay.json']+[Path(m.__file__).resolve() for m in mods];sources={str(x):sha(x) for x in files};need(all(sources[str(Path(m.__file__).resolve())]==p['source_hashes'][str(Path(m.__file__).resolve())] for m in mods),'source drift');recovery_adapter.initialize();need(torch.cuda.is_available() and torch.__version__==p['runtime']['torch'] and torch.version.cuda==p['runtime']['cuda'] and torch.cuda.get_device_name()==p['runtime']['gpu'],'runtime');OUT.mkdir(exist_ok=True);need(not (OUT/'receipt.json').exists() and not list(OUT.glob('*.partial')),'existing receipt/partial');start=time.monotonic()
  with torch.no_grad():model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  model.config.use_cache=False;model.model.config.use_cache=False;mh=probe_eight.model_hash(model);need(mh==p['model_state_sha256'],'model');params=lambda:[(n,id(v),v.data_ptr(),v._version) for n,v in list(model.named_parameters())+list(model.named_buffers())];versions=params()
  def guard():need(params()==versions,'model mutation')
  n=0
  for ident in ids:
   guard();key=hashlib.sha256(ident.encode()).hexdigest()
   if validate(ident,sources):n+=1;continue
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();m,t=e3_source_question.run_case(model,'train',ident,ledger[ident],cat[ident],prefix,guard);torch.cuda.synchronize();m['peak_allocated_bytes']=torch.cuda.max_memory_allocated();m['peak_reserved_bytes']=torch.cuda.max_memory_reserved();tmp=OUT/(key+'.'+uuid.uuid4().hex+'.partial');tp=OUT/(key+'.safetensors')
   try:
    save_file(t,str(tmp));back=load_file(str(tmp));need(set(back)==set(t) and all(torch.equal(back[k],t[k]) for k in t),'readback');os.link(tmp,tp)
   finally:
    if tmp.exists():tmp.unlink()
   m['tensor_sha256']=sha(tp);m['tensor_bytes']=tp.stat().st_size;m['source_hashes']=sources;probe_eight.write_json(OUT/(key+'.json'),m);need(validate(ident,sources),'saved verify');n+=1;print('E3_SOURCE64_CASE',n,ident,flush=True)
  need(n==64 and all(sha(Path(k))==v for k,v in sources.items()) and probe_eight.model_hash(model)==mh,'final source');probe_eight.write_json(OUT/'receipt.json',{'status':'E3_SOURCE64_COLLECTED_PENDING_INDEPENDENT_AUDIT','count':n,'selection_sha256':sha(SEL),'source_hashes':sources,'model_state_sha256':mh,'runtime':p['runtime'],'elapsed_seconds':time.monotonic()-start,'questions':[{'id':ident,'metadata_sha256':sha(OUT/(hashlib.sha256(ident.encode()).hexdigest()+'.json')),'tensor_sha256':sha(OUT/(hashlib.sha256(ident.encode()).hexdigest()+'.safetensors'))} for ident in ids],'scope':'Input-region K/V source perturbations are full-shape SDPA with K1 carrier; QKV streamed and not archived; ID old corner matches. Old ARC outcomes exposed; stratified 64 not population representative.'});print('E3_SOURCE64_PENDING_AUDIT',flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
