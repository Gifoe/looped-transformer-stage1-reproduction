"""4090 local N/P/PK three-case source, no-hook and independent bare technical gate."""
import os,json,hashlib,fcntl,time,pathlib,torch,transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from writer4090_fresh_reference import capture_case
from writer4090_bare_hook import bare,bits
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=pathlib.Path('/root/autodl-tmp/ouro-e1-e3-offload-20261113'); OLD=pathlib.Path('/root/autodl-tmp/ouro-e4-e5'); MODEL=OLD/'model'; INPUT=OLD/'inputs/runner_inputs.jsonl'; PREFIX=OLD/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/natural_prefix_identity_20261111/reference_prefix_ids.json'; ORDER=ROOT/'orders/ORDER_Q_TO_R_INTERVAL_DISCRIMINATION_v1.md';AMEND=ROOT/'orders/AMENDMENT_INTERVAL_PRIMARY_4090_v1.md';BOUND=ROOT/'inputs/boundary_compact.jsonl'; OUT=ROOT/'results/technical_reference_three'; IDS=['ACTAAP_2013_7_16','Mercury_405804','MSA_2012_8_30']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 assert sha(MODEL/'model.safetensors')=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af' and sha(ORDER)=='d0fc019924c6cb1f41a927b6d86bfa6b7e298ed2b90add4d27d7134d03a2349c' and sha(AMEND)=='1c331f1f4ec3f1f12bf00f081d2f79ddae486d1d27d646f6f04f8cfba00c0e2f' and sha(BOUND)=='73c3ddac05dbc4342015163061fe90ac104c653f183b36aa35ffbb38e8234461' and sha(PREFIX)=='e984a118a098923ba87fab4ddd4a26d1f9da8c5dc9767c5ab073352cd3dfbf67'
 rows=[json.loads(x) for x in INPUT.read_text().splitlines()]; qs={q['id']:q for q in rows if q['split']=='test' and q['id'] in IDS};assert set(qs)==set(IDS);prefix=json.loads(PREFIX.read_text())['input_ids'];assert len(prefix)==1839;bound={x['id']:x for line in BOUND.read_text().splitlines() if (x:=json.loads(line))['split']=='test' and x['id'] in IDS};assert set(bound)==set(IDS) and all(bound[i]['input_sha256']==qs[i]['input_ids_sha256'] and bound[i]['readout_position']==qs[i]['R'] and bound[i]['full_boundary_verified'] and bound[i]['prefix_count']==1839 for i in IDS)
 files=[pathlib.Path(__file__).resolve(),ORDER,AMEND,BOUND,PREFIX,INPUT,MODEL/'model.safetensors',MODEL/'config.json',MODEL/'configuration_ouro.py',MODEL/'modeling_ouro.py']+sorted((ROOT/'source').glob('*.py'));sources={str(p):sha(p) for p in files};assert not OUT.exists();OUT.mkdir(parents=True);write(OUT/'STARTED.json',{'status':'STARTED','source_sha256':sources,'site':'4090','parent_order_sha256':sha(ORDER),'site_amendment_sha256':sha(AMEND)})
 lock=os.open(OLD/'gpu.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);m=None;old=None
 try:
  fcntl.flock(lock,fcntl.LOCK_EX);start=time.monotonic();transformers.utils.import_utils._torchvision_available=False;m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();m.config.use_cache=False;m.model.config.use_cache=False;old=m.model.total_ut_steps;m.model.total_ut_steps=2;version=lambda:[(n,id(x),x.data_ptr(),x._version) for n,x in list(m.named_parameters())+list(m.named_buffers())];v0=version();guard=lambda:None if version()==v0 else (_ for _ in ()).throw(ValueError('model mutation'));cases=[]
  for qid in IDS:
   q=qs[qid];x=torch.tensor([q['ids']],device='cuda');R=q['R'];torch.cuda.synchronize();t=time.monotonic()
   with torch.inference_mode():
    nohook=m(x,labels=None,use_cache=False,exit_at_step=1).logits[:,R,:].clone();meta,saved,transient=capture_case(m,q,prefix,guard);assert bits(nohook,saved['N_logits'].cuda())
    with bare(m,8,R,len(prefix)) as n:N=m(x,labels=None,use_cache=False,exit_at_step=1).logits[:,R,:]
    with bare(m,8,R,len(prefix),hidden=n['donor_hidden']) as p:P=m(x,labels=None,use_cache=False,exit_at_step=1).logits[:,R,:]
    with bare(m,8,R,len(prefix),hidden=n['donor_hidden'],key=n['k']) as k:PK=m(x,labels=None,use_cache=False,exit_at_step=1).logits[:,R,:]
    for name,z,rec in [('N',N,n),('P',P,p),('PK',PK,k)]:assert bits(z,saved[name+'_logits'].cuda()) and all(bits(rec[tag+'_R'],saved[name+'_'+tag+'_R'].cuda()) for tag in ('x','a','sdpa'))
   torch.cuda.synchronize();meta['elapsed_seconds_cuda_synchronized']=time.monotonic()-t;stem=hashlib.sha256(qid.encode()).hexdigest();tp=OUT/(stem+'.safetensors');save_file(saved,str(tp));meta['tensor_sha256']=sha(tp);meta['identity_nohook_and_independent_bare_n_p_pk']='BITWISE';mp=OUT/(stem+'.json');write(mp,meta);cases.append({'id':qid,'metadata_sha256':sha(mp),'tensor_sha256':sha(tp)});print('4090_REFERENCE_TECH',qid,meta['elapsed_seconds_cuda_synchronized'],flush=True)
  assert version()==v0 and all(sha(pathlib.Path(k))==v for k,v in sources.items());write(OUT/'RECEIPT.json',{'status':'4090_FRESH_THREE_REFERENCE_BARE_PASSED','cases':cases,'source_sha256':sources,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'elapsed_seconds':time.monotonic()-start,'scope':'4090 no-hook natural K2 compared with observer and independently implemented bare N/P/PK; no A6000 logits used and no treatment effects.'});print('4090_REFERENCE_TECH_PASS',sha(OUT/'RECEIPT.json'),flush=True)
 finally:
  if m is not None and old is not None:m.model.total_ut_steps=old
  os.close(lock)
if __name__=='__main__':main()
