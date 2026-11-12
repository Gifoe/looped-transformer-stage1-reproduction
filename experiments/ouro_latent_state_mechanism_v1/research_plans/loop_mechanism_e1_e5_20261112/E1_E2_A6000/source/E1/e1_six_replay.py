"""Independent CUDA scoring of six captured old-corner suffixes, with old anchors."""
import os,fcntl,json,hashlib,torch
from pathlib import Path
from safetensors.torch import load_file
from e1_donor_pool import OLD,sha
B=Path('/home/luansiyu/loop-latent/repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/E1/six_corner_probe');P=B/'pending.json'; fd=os.open('/home/luansiyu/loop-latent/ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);psha=sha(P);p=json.loads(P.read_text());assert p['status']=='E1_SIX_CORNER_PENDING_REPLAY' and p['count']==3 and all(sha(Path(k))==v for k,v in p['source_hashes'].items()) and torch.cuda.is_available() and torch.__version__==p['runtime']['torch'] and torch.version.cuda==p['runtime']['cuda'] and torch.cuda.get_device_name()==p['runtime']['gpu'];out=[]
for i in range(3):
 mp=B/(str(i)+'.json');tp=mp.with_suffix('.safetensors');assert sha(mp)==p['questions'][i]['metadata_sha256'] and sha(tp)==p['questions'][i]['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));key=hashlib.sha256(m['id'].encode()).hexdigest();oldpath=OLD/'test'/(key+'.safetensors');assert sha(oldpath)==m['old_eight_tensor_sha256'];old=load_file(str(oldpath))
 for n in ('G111','G112','G121','G211','G212','G221'):
  z=t[n+'_logits'];a=t[n+'_attention_localnorm'];sv=t[n+'_injected_sdpa_R'];assert z.dtype==torch.bfloat16 and tuple(z.shape)==(1,49152) and torch.isfinite(z).all() and torch.equal(z,old[n+'_logits']) and torch.equal(a,old[n+'_a_R']) and torch.equal(sv,old[n+'_sdpa_R']);lp=z.cuda().float().log_softmax(-1);scores=[float(lp.gather(-1,torch.tensor([[j]],device='cuda')).sum()) for j in m['targets']];assert scores==m['conditions'][n]['scores']
 out.append({'id':m['id'],'metadata_sha256':sha(mp),'tensor_sha256':sha(tp),'conditions':6})
assert sha(P)==psha and all(sha(Path(k))==v for k,v in p['source_hashes'].items());r={'status':'E1_SIX_CORNER_CUDA_SCORE_AND_OLD_ANCHOR_REPLAY_PASSED','pending_sha256':psha,'count':3,'questions':out,'scope':'Independent CUDA score replay against saved full logits, not another model forward.'};path=B/'replay.json'
with path.open('x') as f:json.dump(r,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('E1_SIX_REPLAY_PASSED',sha(path),flush=True);os.close(fd)
