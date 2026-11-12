"""Review-only natural branch three-question probe. GPU ENTRY DISABLED."""
import json,hashlib,os,fcntl,time,shutil,torch
from pathlib import Path
from safetensors.torch import save_file,load_file
from transformers import AutoModelForCausalLM
from source_input import ROOT,initial_sources,load_one,sha
from technical import verify
from decompose import decompose

def run(out):
 out=Path(out).resolve();out.relative_to(ROOT/'research_plans');out.mkdir(exist_ok=False)
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info()[0]<=20*1024**3 or shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('resource preflight')
  sources=initial_sources();here=Path(__file__).resolve().parent
  for name in ('state_capture.py','technical.py','decompose.py','source_input.py','probe_draft.py'):sources['code_'+name]=here/name
  start={k:sha(v) for k,v in sources.items()};cases={k:load_one(k) for k in ('A','B','C')}
  config=json.loads(sources['config'].read_text());eps=config['rms_norm_eps']
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
  with torch.no_grad():model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  model.config.use_cache=False;model.model.config.use_cache=False
  if model.training or model.config._attn_implementation!='sdpa' or next(model.parameters()).dtype!=torch.bfloat16:raise ValueError('model mode')
  versions=[(id(t),t.data_ptr(),t._version) for t in list(model.parameters())+list(model.buffers())]
  runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'attention':model.config._attn_implementation,'dtype':str(next(model.parameters()).dtype),'flash_sdp':torch.backends.cuda.flash_sdp_enabled(),'mem_efficient_sdp':torch.backends.cuda.mem_efficient_sdp_enabled(),'math_sdp':torch.backends.cuda.math_sdp_enabled()}
  receipts=[]
  for label in ('A','B','C'):
   if torch.cuda.mem_get_info()[0]<4*1024**3 or shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('resource reserve')
   if versions!=[(id(t),t.data_ptr(),t._version) for t in list(model.parameters())+list(model.buffers())]:raise ValueError('model state changed')
   e=cases[label];ids=torch.tensor([e['ids']],dtype=torch.long,device='cuda')
   obs,logits=verify(model,ids,e['R'],e['targets'],e['states'],e['norms'],e['logits'],e['scores'])
   if any(v.dtype!=torch.bfloat16 for item in obs['states'].values() for v in item.values()):raise ValueError('state dtype')
   rows=model.lm_head.weight[e['targets']].detach().float().cpu().numpy();gamma=model.model.norm.weight.detach().float().cpu().numpy();lens=decompose(obs['states'],rows,gamma,eps,list(range(len(e['targets']))),e['gold'])
   tensors={f'u{u}_layer{i:02d}_{k}':v.detach().cpu().contiguous().clone() for (u,i),d in obs['states'].items() for k,v in d.items()}
   tensors.update({f'loop{u+1}_logits':z.detach().cpu().contiguous().clone() for u,z in enumerate(logits)});tensors.update({f'loop{u+1}_post_norm':z.detach().cpu().contiguous().clone() for u,z in enumerate(obs['norms'])})
   tp=out/(label+'.safetensors');save_file(tensors,str(tp));read=load_file(str(tp))
   if set(read)!=set(tensors) or any(not torch.equal(read[k],v) or read[k].dtype!=v.dtype for k,v in tensors.items()):raise ValueError('save/readback')
   meta={'id':e['id'],'label':label,'R_absolute':e['R'],'R_saved_row':e['saved_row'],'targets':e['targets'],'gold':e['gold'],'prior_scores':e['scores'],'natural_metadata_sha256':e['metadata_sha256'],'natural_tensor_sha256':e['tensor_sha256'],'probe_tensor_sha256':sha(tp),'attention':obs['attention'],'lens':lens,'kernel':'UNKNOWN','source_hashes':start,'runtime':runtime,'status':'TECHNICAL_CASE_PENDING_INDEPENDENT_REVIEW'}
   with (out/(label+'.json')).open('x') as f:json.dump(meta,f,sort_keys=True,allow_nan=False);f.write('\n')
   receipts.append({'id':e['id'],'metadata_sha256':sha(out/(label+'.json')),'tensor_sha256':sha(tp)})
  if {k:sha(v) for k,v in sources.items()}!=start or versions!=[(id(t),t.data_ptr(),t._version) for t in list(model.parameters())+list(model.buffers())]:raise ValueError('source drift')
  torch.cuda.synchronize();receipt={'status':'THREE_CASE_TECHNICAL_PENDING_ASTRA_REVIEW','cases':receipts,'elapsed_seconds':time.monotonic()-started,'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'peak_cuda_reserved':torch.cuda.max_memory_reserved(),'source_hashes':start,'runtime':runtime}
  with (out/'receipt.json').open('x') as f:json.dump(receipt,f,sort_keys=True,allow_nan=False);f.write('\n')
 finally:os.close(lock)
if __name__=='__main__':raise SystemExit('DISABLED: Astra code review and supervised launch required')
