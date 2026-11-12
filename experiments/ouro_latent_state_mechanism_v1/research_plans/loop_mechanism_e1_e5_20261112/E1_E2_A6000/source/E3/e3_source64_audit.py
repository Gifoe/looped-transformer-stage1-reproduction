"""Independently recapture natural QKV and replay every saved source64 SDPA/control and CUDA score."""
import os,json,gzip,hashlib,fcntl,torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
import recovery_adapter,probe_eight,e1_question,e3_source_control_v2,eight_observer,eight_runner,natural_l18_hook,technical
from e3_source_control_v2 import build
ROOT=Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E3';OUT=E/'source_train64';P=OUT/'receipt.json';SEL=B/'orders/E1_DISCOVERY64_FROZEN.json'
if not __debug__:raise RuntimeError('optimized Python forbidden')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def thash(x):return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def need(x,s):
 if not x:raise ValueError(s)
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.device==b.device and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
@torch.inference_mode()
def audit_one(model,ident,m,t):
 src,_,_=recovery_adapter.load('train',ident);e=src['natural'];old,ot,_=e1_question.read_old('train',ident);need(m['id']==ident and m['split']=='train' and m['targets']==old['targets']==e['targets'] and m['R']==old['R']==e['R'] and m['input_ids_sha256']==old['input_ids_sha256'],'source identity');ids=torch.tensor([e['ids']],device='cuda');R=e['R'];core=model.model;prior=core.total_ut_steps;caps={}
 try:
  for depth in (1,2):
   core.total_ut_steps=depth;own=src['states'][depth-1]
   with eight_observer.observe_qkv(model,depth,R) as rec,natural_l18_hook.intervene(model,depth,R,own['x'],own['a']) as ar:y=model(ids,labels=None,use_cache=False,exit_at_step=depth-1)
   z=y.logits[0,R,:].unsqueeze(0);cap=eight_runner.snapshot(rec);need(eight_runner.descriptors(cap)==m['captures'][str(depth)]==old['capture_receipts'][str(depth)]['qkv'] and bits(z,e['logits'][depth-1].cuda()) and technical.score(z,e['targets'])==e['scores'][depth-1] and bits(ar['receiver_a'],own['a'].cuda()),'natural recapture');caps[depth]=cap
  names={'ID','K_Q','V_Q','KV_Q','KV_P','K_Q_NORM','V_Q_NORM','V_Q_ROLL'};need(set(m['conditions'])=={'1','2'} and all(set(m['conditions'][str(q)])==names for q in (1,2)),'conditions');expected={f'q{q}_{name}_{suffix}' for q in (1,2) for name in names if m['conditions'][str(q)][name]['control']['status']=='OK' for suffix in ('logits','sdpa_R','a_R')};need(set(t)==expected,'tensor keys');n=0
  for q in (1,2):
   qm=caps[1]['q'].clone()
   if q==2:qm[...,R,:]=caps[2]['q'][...,R,:]
   for name in sorted(names):
    k,v,c=build(caps[1]['k'],caps[2]['k'],caps[1]['v'],caps[2]['v'],R,1839,name);cm=m['conditions'][str(q)][name];need(c==cm['control'],'control recompute')
    if k is None:continue
    need(thash(k)==cm['k_sha256'] and thash(v)==cm['v_sha256'] and thash(qm)==cm['q_sha256'],'source input bytes');s=F.scaled_dot_product_attention(qm,k,v,*caps[1]['tail'],**caps[1]['kwargs'])[...,R,:];tag=f'q{q}_{name}';z=t[tag+'_logits'];sr=t[tag+'_sdpa_R'];ar=t[tag+'_a_R'];need(bits(s,sr.cuda()) and z.dtype==sr.dtype==ar.dtype==torch.bfloat16 and tuple(z.shape)==(1,49152) and tuple(sr.shape)==(1,16,128) and tuple(ar.shape)==(1,2048) and all(torch.isfinite(v).all() for v in (z,sr,ar)) and thash(z)==cm['logits_sha256'] and thash(sr)==cm['sdpa_R_sha256'] and thash(ar)==cm['a_R_sha256'] and cm['outside_R_unchanged'],'actual SDPA');lp=z.cuda().float().log_softmax(-1);sc=[float(lp.gather(-1,torch.tensor([[x]],device='cuda')).sum()) for x in e['targets']];need(sc==cm['scores'],'CUDA scores')
    if name=='ID':need(torch.equal(z,ot[f'G{q}22_logits']) and torch.equal(sr,ot[f'G{q}22_sdpa_R']) and torch.equal(ar,ot[f'G{q}22_a_R']) and sc==old['scores'][f'G{q}22'],'old ID')
    n+=1
  return n
 finally:core.total_ut_steps=prior

def main():
 fd=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);psha=sha(P);r=json.loads(P.read_text());sel=json.loads(SEL.read_text());ids=[v['id'] for g in ('rescue','damage','stable_correct','stable_wrong') for v in sel['groups'][g]];ledger={x['id']:x for x in r['questions']};need(r['status']=='E3_SOURCE64_COLLECTED_PENDING_INDEPENDENT_AUDIT' and r['count']==len(ids)==len(ledger)==64 and set(ledger)==set(ids) and r['selection_sha256']==sha(SEL) and all(sha(Path(k))==v for k,v in r['source_hashes'].items()),'receipt');need(all(str(Path(mod.__file__).resolve()) in r['source_hashes'] and sha(Path(mod.__file__).resolve())==r['source_hashes'][str(Path(mod.__file__).resolve())] for mod in (recovery_adapter,probe_eight,e1_question,e3_source_control_v2,eight_observer,eight_runner,natural_l18_hook,technical)),'actual imported source binding');need(torch.cuda.is_available() and torch.__version__==r['runtime']['torch'] and torch.version.cuda==r['runtime']['cuda'] and torch.cuda.get_device_name()==r['runtime']['gpu'],'runtime');recovery_adapter.initialize()
  with torch.no_grad():model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  model.config.use_cache=False;model.model.config.use_cache=False;mh=probe_eight.model_hash(model);need(mh==r['model_state_sha256'],'model');out=[]
  for ident in ids:
   key=hashlib.sha256(ident.encode()).hexdigest();mp=OUT/(key+'.json');tp=mp.with_suffix('.safetensors');need(sha(mp)==ledger[ident]['metadata_sha256'] and sha(tp)==ledger[ident]['tensor_sha256'],'shard');m=json.loads(mp.read_text());t=load_file(str(tp));need(m['source_hashes']==r['source_hashes'],'source');n=audit_one(model,ident,m,t);out.append({'id':ident,'metadata_sha256':sha(mp),'tensor_sha256':sha(tp),'conditions_replayed':n})
   if len(out)%16==0:print('E3_SOURCE64_AUDITED',len(out),flush=True)
  need(sha(P)==psha and all(sha(Path(k))==v for k,v in r['source_hashes'].items()) and probe_eight.model_hash(model)==mh,'final source');result={'status':'E3_SOURCE64_NATURAL_RECAPTURE_SDPA_CUDA_SCORE_AUDIT_PASSED','count':len(out),'collector_receipt_sha256':psha,'selection_sha256':sha(SEL),'script_sha256':sha(Path(__file__)),'questions':out,'scope':'Separate process recaptures natural K1/K2 using same frozen observer, rebuilds full-shape SDPA source controls, checks saved SDPA_R bitwise and independently replays saved final logits scores; not an independent implementation of model suffix.'};path=E/'source64_independent_audit.json'
  with path.open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  print('E3_SOURCE64_AUDIT_COMPLETE',sha(path),flush=True)
 finally:os.close(fd)
if __name__=='__main__':main()
