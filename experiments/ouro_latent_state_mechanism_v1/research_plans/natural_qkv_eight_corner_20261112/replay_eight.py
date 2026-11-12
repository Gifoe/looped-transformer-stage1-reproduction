"""Independent saved CUDA score + old four-corner replay."""
import os,sys,json,hashlib,torch
from pathlib import Path
from safetensors.torch import load_file
from technical import score
from state_capture import bits
import recovery_adapter
from source_input import sha
N={f'G{i}{j}{k}' for i in (1,2) for j in (1,2) for k in (1,2)}
def need(x,s):
 if not x:raise ValueError(s)
def thash(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def main():
 out=Path(sys.argv[1]);pending=json.loads((out/'probe_pending.json').read_text());need(pending['status']=='PENDING_INDEPENDENT_CUDA_REPLAY' and len(pending['questions'])==3,'pending');need(all(pending['runtime'][k]==v for k,v in {'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()}.items()),'runtime');recovery_adapter.initialize();results=[]
 for row in pending['questions']:
  label=row['label'];mp=out/(label+'.json');tp=out/(label+'.safetensors');need(sha(mp)==row['metadata_sha256'] and sha(tp)==row['tensor_sha256'],'hash');m=json.loads(mp.read_text());t=load_file(str(tp));src,old,oldp=recovery_adapter.load(m['split'],m['id']);e=src['natural'];ot=load_file(str(oldp));need(set(m['scores'])==N and set(t)=={f'{n}_{suffix}' for n in N for suffix in ('logits','sdpa_R','a_R')} and m['targets']==e['targets'] and m['gold']==e['gold'] and m['R']==e['R'] and m['input_ids_sha256']==hashlib.sha256(json.dumps(e['ids']).encode()).hexdigest(),'request/keys')
  for n in N:
   cr=m['corner_receipts'][n];shape=cr['actual_receiver_qkv']['qkv']['q']['shape'];z=t[n+'_logits'];sd=t[n+'_sdpa_R'];a=t[n+'_a_R'];need(z.dtype==torch.bfloat16 and tuple(z.shape)==(1,49152) and sd.dtype==torch.bfloat16 and list(sd.shape)==[shape[0],shape[1],shape[3]] and a.dtype==torch.bfloat16 and tuple(a.shape)==(1,2048) and all(torch.isfinite(x).all() for x in (z,sd,a)) and cr['outside_R_unchanged'] is True and cr['root_input_events']==1 and thash(sd)==cr['injected_sdpa_R_sha256'] and thash(a)==cr['a_R_sha256'] and score(z.to('cuda'),e['targets'])==m['scores'][n],'CUDA score/receipt')
  for n,o in [('G111','F11'),('G211','F21'),('G122','F12'),('G222','F22')]:
   need(all(bits(t[n+'_'+suffix],ot[o+'_'+suffix]) for suffix in ('logits','sdpa_R','a_R')) and m['scores'][n]==old['scores'][o],'old four corner parity')
  results.append({'label':label,'id':m['id'],'split':m['split'],'metadata_sha256':sha(mp),'tensor_sha256':sha(tp),'old_metadata_sha256':sha(oldp.with_suffix('.json')),'old_tensor_sha256':sha(oldp)})
 result={'status':'EIGHT_CORNER_THREE_CASE_CUDA_REPLAY_PASSED','pending_sha256':sha(out/'probe_pending.json'),'questions':results,'runtime':pending['runtime'],'script_sha256':sha(Path(__file__))}
 with (out/'replay.json').open('x') as f:json.dump(result,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 print('EIGHT_CORNER_CUDA_REPLAY_PASSED',flush=True)
if __name__=='__main__':main()
