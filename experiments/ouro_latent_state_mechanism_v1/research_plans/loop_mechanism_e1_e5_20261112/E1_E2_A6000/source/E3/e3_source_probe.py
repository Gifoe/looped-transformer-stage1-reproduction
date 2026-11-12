"""Three fixed test examples, E3 source rollback technical probe (do not interpret effects)."""
import os,json,gzip,hashlib,time,fcntl,resource,torch
from pathlib import Path
from safetensors.torch import save_file,load_file
from transformers import AutoModelForCausalLM
import recovery_adapter,probe_eight,e1_question,e1_injection_observer,e3_source_control_v2,e3_source_question,eight_observer,eight_runner,qkv_runner,natural_l18_hook,natural_sources,technical,full_input,source_input,state_capture,replay_eight
ROOT=Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';OUT=B/'E3/source_technical';P=B/'E1/technical_probe';LED=ROOT/'research_plans/prompt_boundary_audit_20261111/boundary_ledger.jsonl';COV=ROOT/'research_plans/prefix_question_partition_draft_20261111/saved_offset_coverage.json';REF=ROOT/'research_plans/natural_prefix_identity_20261111/reference_prefix_ids.json';CAT=B/'results/compact_input_catalog.jsonl.gz';ORDER=B/'orders/ORDER_E3_SOURCE_INTERVENTION_v1.md';SEL=['ACTAAP_2013_7_16','Mercury_405804','MSA_2012_8_30']
if not __debug__:raise RuntimeError('optimized Python forbidden')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def need(x,s):
 if not x:raise ValueError(s)
def main():
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);need(not OUT.exists(),'existing output');OUT.mkdir();p=json.loads((P/'pending.json').read_text());replay=json.loads((P/'replay.json').read_text());need(replay['pending_sha256']==sha(P/'pending.json') and replay['status']=='E1_THREE_CASE_CUDA_SCORE_AND_ENDPOINT_REPLAY_PASSED' and all(sha(Path(k))==v for k,v in p['source_hashes'].items()),'E1 technical source');need(sha(ORDER)=='07f9e62111beda94f0b856c96967862ef7bcb24f09a80d043c38eac806b89a94' and sha(LED)=='a77258ade36415efb17ca09a886417f9b265ce53feec1589431dff6062f4e2bc' and sha(COV)=='5263cc8bf07900822b11954194b7e9fd60c3066c9f43d96baee092157445dba6' and sha(REF)=='e984a118a098923ba87fab4ddd4a26d1f9da8c5dc9767c5ab073352cd3dfbf67' and sha(CAT)=='a38218bfe0c9940f326663b8ecc0447f875a18c07f0091813ddfef4a3d12342d','boundary/selection');
  with LED.open() as f:ledger={r['id']:r for line in f if (r:=json.loads(line))['split']=='test' and r['id'] in SEL}
  with gzip.open(CAT,'rt') as f:cat={r['id']:r for line in f if (r:=json.loads(line))['split']=='test' and r['id'] in SEL}
  prefix=json.loads(REF.read_text())['input_ids'];need(len(prefix)==1839 and set(ledger)==set(cat)==set(SEL),'cohort');mods=(recovery_adapter,probe_eight,e1_question,e1_injection_observer,e3_source_control_v2,e3_source_question,eight_observer,eight_runner,qkv_runner,natural_l18_hook,natural_sources,technical,full_input,source_input,state_capture,replay_eight);files=[Path(__file__).resolve(),Path(__file__).with_name('e3_source_replay.py'),Path(__file__).with_name('test_e3_source_control_v2.py'),Path(__file__).with_name('test_e3_source_suffix_oracle.py'),ORDER,LED,COV,REF,CAT,P/'pending.json',P/'replay.json']+[Path(m.__file__).resolve() for m in mods];sources={str(x):sha(x) for x in files};need(all(sha(Path(m.__file__).resolve())==p['source_hashes'][str(Path(m.__file__).resolve())] for m in (recovery_adapter,probe_eight,e1_injection_observer,eight_observer,eight_runner,qkv_runner,natural_l18_hook,natural_sources,technical,full_input,source_input,state_capture,replay_eight)),'shared source drift');ck=ROOT/'model/model.safetensors';need(sha(ck)=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af','checkpoint');recovery_adapter.initialize();need(torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.__version__==p['runtime']['torch'] and torch.version.cuda==p['runtime']['cuda'] and torch.cuda.get_device_name()==p['runtime']['gpu'],'runtime');start=time.monotonic();torch.cuda.reset_peak_memory_stats()
  with torch.no_grad():model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  model.config.use_cache=False;model.model.config.use_cache=False;mh=probe_eight.model_hash(model);need(mh==p['model_state_sha256'],'model state');versions=lambda:[(n,id(v),v.data_ptr(),v._version) for n,v in list(model.named_parameters())+list(model.named_buffers())];v0=versions()
  def guard():need(versions()==v0,'model changed')
  for i,ident in enumerate(SEL):
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();m,t=e3_source_question.run_case(model,'test',ident,ledger[ident],cat[ident],prefix,guard);torch.cuda.synchronize();m['peak_allocated_bytes']=torch.cuda.max_memory_allocated();m['peak_reserved_bytes']=torch.cuda.max_memory_reserved();tp=OUT/(str(i)+'.safetensors');save_file(t,str(tp));back=load_file(str(tp));need(set(t)==set(back) and all(torch.equal(t[k],back[k]) for k in t),'readback');m['tensor_sha256']=sha(tp);m['tensor_bytes']=tp.stat().st_size;probe_eight.write_json(OUT/(str(i)+'.json'),m);print('E3_SOURCE_TECH_CASE',i,ident,m['question_wall_seconds_cuda_synchronized_excludes_save'],m['tensor_bytes'],flush=True)
  guard();need(probe_eight.model_hash(model)==mh and sha(ck)=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af' and all(sha(Path(k))==v for k,v in sources.items()),'final drift');probe_eight.write_json(OUT/'pending.json',{'status':'E3_SOURCE_THREE_CASE_PENDING_REPLAY','count':3,'source_hashes':sources,'model_state_sha256':mh,'runtime':p['runtime'],'elapsed_seconds':time.monotonic()-start,'rss_peak_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'questions':[{'metadata_sha256':sha(OUT/(str(i)+'.json')),'tensor_sha256':sha(OUT/(str(i)+'.safetensors'))} for i in range(3)]});print('E3_SOURCE_TECH_PENDING_REPLAY',flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
