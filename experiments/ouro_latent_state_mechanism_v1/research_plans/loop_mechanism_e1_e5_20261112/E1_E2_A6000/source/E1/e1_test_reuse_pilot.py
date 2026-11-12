"""Fixed three-case independent output parity for duplicate-verification removal; no test treatment results interpreted."""
import os,json,time,hashlib,fcntl,torch
from pathlib import Path
from transformers import AutoModelForCausalLM
import recovery_adapter,probe_eight,e1_question,e1_six_question,e1_six_question_reuse_v2,e1_donor_pool,technical
R=Path('/home/luansiyu/loop-latent');B=R/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E1';OUT=E/'test_reuse_pilot';SEL=['ACTAAP_2013_7_16','Mercury_405804','MSA_2012_8_30']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def need(x,s):
 if not x:raise ValueError(s)
def main():
 lock=os.open(R/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX);p=json.loads((E/'technical_probe/pending.json').read_text());r=json.loads((E/'technical_probe/replay.json').read_text());need(r['status']=='E1_THREE_CASE_CUDA_SCORE_AND_ENDPOINT_REPLAY_PASSED' and r['pending_sha256']==sha(E/'technical_probe/pending.json') and all(sha(Path(k))==v for k,v in p['source_hashes'].items()),'technical proof');mods=(recovery_adapter,probe_eight,e1_question,e1_six_question,e1_six_question_reuse_v2,e1_donor_pool,technical);amend=B/'orders/AMENDMENT_E1_TEST_REUSE_VERIFIED_NATURAL_v2.md';need(sha(amend)=='81c81d120a2336bb110da2843bc562aeb967fcbba9ba92bdbf8ea68544dfccb7','amendment');trp=E/'train/remainder_receipt.json';tr=json.loads(trp.read_text());ta=json.loads((E/'remainder_cuda_audit.json').read_text());need(ta['collector_receipt_sha256']==sha(trp) and ta['status']=='TRAIN_REMAINDER_INDEPENDENT_CUDA_SCORE_AND_SOURCE_AUDIT_PASSED','train source proof');files=[Path(__file__).resolve(),amend,trp,E/'remainder_cuda_audit.json',E/'technical_probe/pending.json',E/'technical_probe/replay.json']+[Path(m.__file__).resolve() for m in mods];sources={str(x):sha(x) for x in files};need(all(sources[str(Path(m.__file__).resolve())]==p['source_hashes'][str(Path(m.__file__).resolve())] for m in (recovery_adapter,probe_eight,technical)) and all(sources[str(Path(m.__file__).resolve())]==tr['sources'][str(Path(m.__file__).resolve())] for m in (e1_question,e1_six_question)),'prior source');recovery_adapter.initialize();need(torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.__version__==p['runtime']['torch'] and torch.version.cuda==p['runtime']['cuda'] and torch.cuda.get_device_name()==p['runtime']['gpu'],'runtime');need(not OUT.exists(),'existing output');OUT.mkdir();start=time.monotonic()
  with torch.no_grad():m=AutoModelForCausalLM.from_pretrained(R/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  m.config.use_cache=False;m.model.config.use_cache=False;mh=probe_eight.model_hash(m);need(mh==p['model_state_sha256'],'model');v0=[(n,id(v),v.data_ptr(),v._version) for n,v in list(m.named_parameters())+list(m.named_buffers())];guard=lambda: need([(n,id(v),v.data_ptr(),v._version) for n,v in list(m.named_parameters())+list(m.named_buffers())]==v0,'model mutation');pool=e1_donor_pool.Pool();out=[]
  for ident in SEL:
   shared,_=e1_question.run_case(m,ident,pool,guard,'test',include_restore=False,save_kv=False);torch.cuda.synchronize();t=time.monotonic();old,ot=e1_six_question.one(m,ident,guard,'test');torch.cuda.synchronize();oldtime=time.monotonic()-t;t=time.monotonic();new,nt=e1_six_question_reuse_v2.one(m,ident,guard,'test',shared);torch.cuda.synchronize();newtime=time.monotonic()-t;old.pop('question_wall_seconds');new.pop('question_wall_seconds');need(old==new and set(ot)==set(nt) and all(torch.equal(ot[k],nt[k]) for k in ot),'complete metadata/tensor parity');out.append({'id':ident,'tensor_keys':len(ot),'old_seconds_cuda_synchronized':oldtime,'reuse_seconds_cuda_synchronized':newtime,'metadata_equal_except_wall':True,'all_tensors_bit_equal':True});print('E1_TEST_REUSE_PILOT_CASE',ident,oldtime,newtime,flush=True)
  need(probe_eight.model_hash(m)==mh and all(sha(Path(k))==v for k,v in sources.items()),'source drift');probe_eight.write_json(OUT/'receipt.json',{'status':'E1_TEST_REUSE_THREE_CASE_FULL_SIX_OUTPUT_PARITY_PASSED','questions':out,'source_hashes':sources,'model_state_sha256':mh,'runtime':p['runtime'],'elapsed_seconds':time.monotonic()-start,'scope':'Same prior verified e1_question native check reused for six old suffix; compares all six-stage output tensors and metadata excluding elapsed time. Other 14 scientific conditions, source anchors, dtype, mask, backend unchanged.'});print('E1_TEST_REUSE_PILOT_PASSED',sha(OUT/'receipt.json'),flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
