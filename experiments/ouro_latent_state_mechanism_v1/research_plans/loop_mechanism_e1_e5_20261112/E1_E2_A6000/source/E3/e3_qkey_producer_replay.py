"""Independent process CUDA scoring replay of saved producer logits (not model-forward replay)."""
import os,json,hashlib,fcntl,torch
from pathlib import Path
from safetensors.torch import load_file
import technical,recovery_adapter,probe_eight
ROOT=Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';OUT=B/'E3/qkey_producer_technical';SEL=['ACTAAP_2013_7_16','Mercury_405804','MSA_2012_8_30']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def thash(x):return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def need(x,s):
 if not x:raise ValueError(s)
def main():
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX); pending=OUT/'pending.json';r=json.loads(pending.read_text());h=sha(pending);need(r['status']=='E3_QKEY_PRODUCER_THREE_CASE_PENDING_REPLAY' and r['count']==3 and len(r['questions'])==3 and all(sha(Path(k))==v for k,v in r['source_hashes'].items()),'source evidence');need(torch.cuda.is_available() and torch.__version__==r['runtime']['torch'] and torch.version.cuda==r['runtime']['cuda'] and torch.cuda.get_device_name()==r['runtime']['gpu'],'runtime');recovery_adapter.initialize();out=[]
  for i,ident in enumerate(SEL):
   mp=OUT/(str(i)+'.json');tp=OUT/(str(i)+'.safetensors');need(sha(mp)==r['questions'][i]['metadata_sha256'] and sha(tp)==r['questions'][i]['tensor_sha256'],'shard');m=json.loads(mp.read_text());t=load_file(str(tp));src,_,_=recovery_adapter.load('test',ident);e=src['natural'];need(m['id']==ident and m['split']=='test' and m['R']==e['R'] and m['targets']==e['targets'] and m['natural_metadata_sha256']==e['metadata_sha256'],'identity');need(set(m['conditions'])=={'0','8','16','17'},'cuts');keys=set()
   for cut in ('0','8','16','17'):
    need(set(m['conditions'][cut])=={'A0K0','A0K1','A1K1','A1K0'},'corners')
    for c,entry in m['conditions'][cut].items():
     tag=f'cut{cut}_{c}';keys.update(tag+'_'+z for z in ('logits','sdpa_R','a_R','receiver_x_R'));z=t[tag+'_logits'];a=t[tag+'_a_R'];sd=t[tag+'_sdpa_R'];xr=t[tag+'_receiver_x_R'];need(z.dtype==a.dtype==sd.dtype==xr.dtype==torch.bfloat16 and z.shape==(1,49152) and a.shape==xr.shape==(1,2048) and sd.ndim==3 and all(torch.isfinite(x).all() for x in (z,a,sd,xr)) and technical.score(z.cuda(),e['targets'])==entry['scores'],'tensor or score');need(entry['outside_R_unchanged'] and thash(z)==entry['logits_sha256'] and thash(a)==entry['a_R_sha256'] and thash(sd)==entry['sdpa_R_sha256'] and thash(xr)==entry['receiver_x_R_sha256'] and entry['mixed_K_Q_dtype']=='torch.bfloat16' and len(entry['mixed_K_Q_shape'])==4,'scope or tensor hash');need((entry['donor_K_Q_sha256'] is None if c in ('A0K0','A1K1') else entry['donor_K_Q_sha256']==entry['mixed_K_Q_sha256']),'mixed key identity')
     if c=='A0K0':need(torch.equal(z,e['logits'][1]) and torch.equal(a,src['states'][1]['a']) and torch.equal(xr,src['states'][1]['x']),'natural endpoint')
   need(set(t)==keys,'tensor key set');out.append({'id':ident,'metadata_sha256':sha(mp),'tensor_sha256':sha(tp),'score_conditions':16})
  need(sha(pending)==h and all(sha(Path(k))==v for k,v in r['source_hashes'].items()),'source drift');probe_eight.write_json(OUT/'replay.json',{'status':'E3_QKEY_PRODUCER_THREE_CASE_CUDA_SCORE_REPLAY_PASSED','pending_sha256':h,'questions':out,'scope':'independent process scoring of saved logits; no independent model forward'})
  print('QKEY_TECH_REPLAY_OK',sha(OUT/'replay.json'),flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
