"""Independent persisted CUDA score/endpoint replay for the 3 QKV probes."""
import json,sys,torch,hashlib,technical
from pathlib import Path
from safetensors.torch import load_file
from technical import score
from natural_sources import load as load_source,sha
from source_input import ROOT
from state_capture import bits
if not __debug__:raise RuntimeError('optimized Python forbidden')
def require(x,s):
 if not x:raise ValueError(s)
def thash(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def main():
 out=Path(sys.argv[1]);pending=json.loads((out/'probe_pending.json').read_text());require(pending['status']=='PENDING_REPLAY' and len(pending['questions'])==3 and all(pending['runtime'][k]==v for k,v in {'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()}.items()) and pending['source_hashes']['code_technical']==sha(Path(technical.__file__)),'pending/runtime/technical')
 result=[]
 for row in pending['questions']:
  mp=out/(row['label']+'.json');tp=out/(row['label']+'.safetensors');require(sha(mp)==row['metadata_sha256'] and sha(tp)==row['tensor_sha256'],'hash');m=json.loads(mp.read_text());t=load_file(str(tp));names={'F11','F21','F12','F22'};require(set(m['scores'])==names and set(t)=={f'{n}_{suffix}' for n in names for suffix in ('logits','sdpa_R','a_R')},'keys');src=load_source(m['split'],m['id']);e=src['natural'];require(m['targets']==e['targets'] and m['gold']==e['gold'] and m['R']==e['R'] and m['input_ids_sha256']==hashlib.sha256(json.dumps(e['ids']).encode()).hexdigest() and m['natural_metadata_sha256']==e['metadata_sha256'] and m['natural_tensor_sha256']==e['tensor_sha256'] and m['branch_metadata_sha256']==src['source_metadata_sha256'] and m['branch_tensor_sha256']==src['source_tensor_sha256'],'source identity');key=m['split']+'_'+__import__('hashlib').sha256(m['id'].encode()).hexdigest();l18=load_file(str(ROOT/'research_plans/natural_l18_full_run_20261111'/(key+'.safetensors')))
  for name in names:
   cr=m['corner_receipts'][name];qshape=cr['actual_receiver_qkv']['qkv']['q']['shape'];sd=t[name+'_sdpa_R'];ar=t[name+'_a_R'];require(len(qshape)==4 and sd.dtype==torch.bfloat16 and list(sd.shape)==[qshape[0],qshape[1],qshape[3]] and thash(sd)==cr['injected_sdpa_R_sha256'] and thash(ar)==cr['a_R_sha256'] and cr['outside_R_unchanged'] is True and cr['root_input_events']==1,'injection receipt')
   z=t[name+'_logits'];require(z.dtype==torch.bfloat16 and z.shape==(1,49152) and torch.isfinite(z).all() and score(z.to('cuda'),m['targets'])==m['scores'][name],'CUDA score');require(t[name+'_a_R'].dtype==torch.bfloat16 and t[name+'_a_R'].shape==(1,2048) and torch.isfinite(t[name+'_a_R']).all() and torch.isfinite(t[name+'_sdpa_R']).all(),'injection tensor')
  require(bits(t['F11_logits'],e['logits'][0]) and bits(t['F22_logits'],l18['T12']) and bits(t['F11_a_R'],src['states'][0]['a']) and bits(t['F22_a_R'],src['states'][1]['a']),'endpoints')
  result.append({'label':row['label'],'id':m['id'],'metadata_sha256':sha(mp),'tensor_sha256':sha(tp)})
 result={'status':'QKV_PROBE_CUDA_REPLAY_PASSED','questions':result,'pending_sha256':sha(out/'probe_pending.json'),'script_sha256':sha(Path(__file__)),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()}
 with (out/'replay.json').open('x') as f:json.dump(result,f,sort_keys=True,allow_nan=False);f.write('\n')
 print('QKV_PROBE_CUDA_REPLAY_PASSED',flush=True)
if __name__=='__main__':main()
