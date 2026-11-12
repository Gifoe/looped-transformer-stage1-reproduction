"""4090 technical three: all fixed 14 branch treatments with local references and donors."""
import os,json,hashlib,fcntl,time,pathlib,torch,transformers
from safetensors.torch import save_file,load_file
from transformers import AutoModelForCausalLM
import writer4090_fresh_reference as refmod, q_to_r_treatment_question as treatments, unrelated_reference_ledger as ledger
from writer4090_model_digest import model_digest
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=pathlib.Path('/root/autodl-tmp/ouro-e1-e3-offload-20261113');OLD=pathlib.Path('/root/autodl-tmp/ouro-e4-e5');MODEL=OLD/'model';INPUT=OLD/'inputs/runner_inputs.jsonl';PREFIX=OLD/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/natural_prefix_identity_20261111/reference_prefix_ids.json';Q=ROOT/'results';TECH=Q/'technical_reference_three';TRAIN=Q/'reference_train64';L=Q/'frozen_unrelated_4090.json';LA=Q/'frozen_unrelated_4090_independent_audit.json';OUT=Q/'treatment_technical_three'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def loadrec(base,c):
 stem=hashlib.sha256(c['id'].encode()).hexdigest();mp=base/(stem+'.json');tp=base/(stem+'.safetensors');assert sha(mp)==c['metadata_sha256'] and sha(tp)==c['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));return m,t
 def_unused=0
def main():
 frozen=json.loads(L.read_text());audit=json.loads(LA.read_text());tr=json.loads((TRAIN/'RECEIPT.json').read_text());te=json.loads((TECH/'RECEIPT.json').read_text());assert frozen['status']=='4090_UNRELATED_SITE_LEDGER_FROZEN_BEFORE_TREATMENTS' and audit['status']=='4090_DONOR_MAPPING_INDEPENDENT_AUDIT_PASSED' and audit['ledger_sha256']==sha(L) and frozen['train_reference_receipt_sha256']==sha(TRAIN/'RECEIPT.json') and frozen['technical_receipt_sha256']==sha(TECH/'RECEIPT.json');assert all(sha(pathlib.Path(k))==v for k,v in tr['source_sha256'].items()) and all(sha(pathlib.Path(k))==v for k,v in te['source_sha256'].items());pool=[]
 for c in tr['cases']:
  m,t=loadrec(TRAIN,c);pool.append({'id':m['id'],'input_ids_sha256':m['input_ids_sha256'],'deltas':{(j,k):t[f'layer{j:02d}_{k}_delta'] for j in range(8,16) for k in ('a','b')}})
 ids=[c['id'] for c in te['cases']];assert ids==['ACTAAP_2013_7_16','Mercury_405804','MSA_2012_8_30'];rows=[json.loads(x) for x in INPUT.read_text().splitlines()];qs={q['id']:q for q in rows if q['split']=='test' and q['id'] in ids};prefix=json.loads(PREFIX.read_text())['input_ids'];files=[pathlib.Path(__file__).resolve(),L,LA,TRAIN/'RECEIPT.json',TECH/'RECEIPT.json',INPUT,PREFIX,MODEL/'model.safetensors',MODEL/'config.json',MODEL/'modeling_ouro.py']+sorted((ROOT/'source').glob('*.py'));sources={str(p):sha(p) for p in files};assert not OUT.exists();OUT.mkdir();write(OUT/'STARTED.json',{'status':'STARTED','source_sha256':sources});lock=os.open(OLD/'gpu.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);m=None;old=None
 try:
  fcntl.flock(lock,fcntl.LOCK_EX);transformers.utils.import_utils._torchvision_available=False;m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();m.config.use_cache=False;m.model.config.use_cache=False;old=m.model.total_ut_steps;ver=lambda:[(n,id(x),x.data_ptr(),x._version) for n,x in list(m.named_parameters())+list(m.named_buffers())];v0=ver();mh=model_digest(m);guard=lambda:None if ver()==v0 else (_ for _ in ()).throw(ValueError('model mutation'));cases=[]
  for c in te['cases']:
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.monotonic();q=qs[c['id']];r,t=loadrec(TECH,c);meta,anchor,ref=refmod.capture_case(m,q,prefix,guard);assert all(k in t and torch.equal(v,t[k]) for k,v in anchor.items());recipient={'id':q['id'],'input_ids_sha256':r['input_ids_sha256'],'deltas':{(j,k):t[f'layer{j:02d}_{k}_delta'] for j in range(8,16) for k in ('a','b')}};mapping,u=ledger.unrelated_directions(recipient,pool);assert mapping==frozen['mapping'][q['id']]['sites'];ids=torch.tensor([q['ids']],device='cuda');conditions,saved=treatments.run_treatments(m,ident=q['id'],ids=ids,R=q['R'],prefix_len=len(prefix),targets=q['targets'],ref=ref,unrelated=u,mapping=mapping,guard=guard);expected={f'{bg}_{verb}_{branch}_{kind}' for branch in ('a','b') for bg,verb,kinds in (('P','plus',('main','norm','unrelated','main_plus_key')),('N','minus',('main','norm','unrelated'))) for kind in kinds};assert set(conditions)==expected;saved.update({k:v for k,v in t.items() if k.startswith(('N_','P_','PK_'))});torch.cuda.synchronize();meta={'question_wall_seconds_cuda_synchronized_excludes_save':time.monotonic()-t0,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'id':q['id'],'split':'test','input_ids_sha256':q['input_ids_sha256'],'R':q['R'],'targets':q['targets'],'conditions':conditions,'reference_tensor_sha256':c['tensor_sha256'],'donor_ledger_sha256':sha(L)};stem=hashlib.sha256(q['id'].encode()).hexdigest();tp=OUT/(stem+'.safetensors');save_file(saved,str(tp));meta['tensor_sha256']=sha(tp);meta['tensor_bytes']=tp.stat().st_size;mp=OUT/(stem+'.json');write(mp,meta);cases.append({'id':q['id'],'metadata_sha256':sha(mp),'tensor_sha256':sha(tp)});print('4090_TREATMENT_TECH',q['id'],flush=True)
  assert ver()==v0 and model_digest(m)==mh and all(sha(pathlib.Path(k))==v for k,v in sources.items());write(OUT/'PENDING.json',{'status':'4090_TREATMENT_THREE_PENDING_BARE','cases':cases,'source_sha256':sources,'model_state_sha256':mh,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda})
 finally:
  if m is not None and old is not None:m.model.total_ut_steps=old
  os.close(lock)
if __name__=='__main__':main()
