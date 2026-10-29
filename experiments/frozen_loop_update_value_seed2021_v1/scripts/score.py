"""All-depth token-logit readouts with causal question probes. No label access in score_view."""
import runtime_bootstrap
import os,sys,json,time,hashlib,random
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent/'looped_stage1_trajectory_audit_seed2021_v1'
CFG=json.loads((ROOT/'configs/protocol.json').read_text())
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HF_HOME=str(ROOT/'cache/huggingface'))
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from lm_eval.api.model import TemplateLM
import importlib.metadata
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(p)
def read(p):
 with (ROOT/p).open(encoding='utf8') as f:return [json.loads(l) for l in f if l.strip()]
def encode(tok,prompt,answer):
 p,t=TemplateLM._encode_pair(SimpleNamespace(tok_encode=lambda s,**kw:tok.encode(s,add_special_tokens=False)),prompt,answer)
 assert p and t and len(p)+len(t)<=CFG['max_context_tokens']
 return p,t
def action_logits(z):
 z1,z2,z3,z4=z;top=z4.softmax(-1).topk(2,dim=-1).values;w=1-(top[:,0]-top[:,1])
 return dict(depth1=z1,depth2=z2,rollback=z3,keep=z4,forward=z4+.5*(z4-z1),adaptive=z4+w[:,None]*(z4-z1),negative=z4-.5*(z4-z1))
def probes(tok,prompt,question,prefix):
 start=prompt.rfind(question);assert start>=0
 enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
 ids=enc['input_ids'];offsets=enc['offset_mapping']
 # Verify token identities against the actual harness-encoded causal prefix.
 good=[i for i,(a,b) in enumerate(offsets) if i>0 and i<len(prefix) and a>=start and b<=start+len(question) and b>a and ids[i]==prefix[i]]
 assert good,'No valid question probe tokens'
 picked=[good[i] for i in np.unique(np.linspace(0,len(good)-1,min(CFG['probe_tokens'],len(good)),dtype=int))]
 assert all(prefix[:i+1]==ids[:i+1] for i in picked)
 return picked
def geometry(raw):
 h=torch.stack(raw).float().cpu().numpy().reshape(4,-1);d=np.diff(h,axis=0)
 norm=lambda x:float(np.linalg.norm(x))
 cos=lambda a,b:float(a@b/max(norm(a)*norm(b),1e-12))
 p=norm(d[2]+d[1])**2/4;s=norm(d[2]-d[1])**2/4
 f=dict(align23=cos(d[0],d[1]),align34=cos(d[1],d[2]),align24=cos(d[0],d[2]),curvature=norm(d[2]-d[1])/max(norm(d[1]),1e-12),persistence=(p-s)/max(p+s,1e-12),norm2=norm(d[0]),norm3=norm(d[1]),norm4=norm(d[2]))
 rng=np.random.default_rng(2021);proj=rng.choice([-1.,1.],size=(d.shape[1],CFG['projection_dim']))/np.sqrt(d.shape[1])
 # Direction projections, not scale-dominated raw magnitudes.
 projected=(d/np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-12))@proj
 return f,projected.flatten().tolist()
def score_view(tok,model,pairs,question,parity_probe=False):
 sums={k:[] for k in CFG['actions']+CFG['diagnostics']};tokvals={k:[] for k in sums};counts=[]
 torch.cuda.synchronize();begin=time.monotonic();probe_loss=None;geom=None;projection=None;stats=None
 for ci,(prompt,answer) in enumerate(pairs):
  prefix,target=encode(tok,prompt,answer);pi=probes(tok,prompt,question,prefix) if ci==0 else []
  positions=[i-1 for i in pi]+list(range(len(prefix)-1,len(prefix)+len(target)-1))
  targets=[prefix[i] for i in pi]+target
  x=torch.tensor([prefix+target[:-1]],device='cuda');raw=[]
  hook=model.model.norm.register_forward_pre_hook(lambda mod,args:raw.append(args[0][:,len(prefix)-1,:].detach().clone())) if ci==0 else None
  try:_,states,_=model.model(input_ids=x,use_cache=False)
  finally:
   if hook:hook.remove()
  assert len(states)==4
  z=[model.lm_head(s[:,positions,:])[0].float() for s in states]
  acts=action_logits(z);t=torch.tensor(targets,device='cuda');npb=len(pi)
  for k,a in acts.items():
   ll=a.log_softmax(-1).gather(-1,t[:,None])[:,0]
   sums[k].append(float(ll[npb:].sum()));tokvals[k].append(ll[npb:].cpu().tolist())
   if ci==0:
    if probe_loss is None:probe_loss={}
    probe_loss[k]=float(-ll[:npb].mean())
  counts.append(len(target))
  if ci==0:
   geom,projection=geometry(raw)
   lp1=z[0][npb].log_softmax(-1);lp4=z[3][npb].log_softmax(-1);p4=lp4.exp();top=p4.topk(2).values
   stats=dict(entropy=float(-(p4*lp4).sum()),confidence=float(top[0]),vocab_margin=float(top[0]-top[1]),kl41=float((p4*(lp4-lp1)).sum()))
   probe_record=dict(token_indices=pi,targets=[prefix[i] for i in pi],positions=[i-1 for i in pi],prefix_tokens=len(prefix))
   if parity_probe:
    # Same causal prediction with all future input removed; no gold used.
    idx=pi[-1];short=torch.tensor([prefix[:idx]],device='cuda')
    _,ss,_=model.model(input_ids=short,use_cache=False)
    check=model.lm_head(ss[-1][:,-1,:])[0].float().log_softmax(-1)[prefix[idx]]
    actual=z[3][npb-1].log_softmax(-1)[prefix[idx]]
    err=abs(float(check-actual));assert err<=.05,err;probe_record['causal_prefix_nll_error']=err
  del x,states,z,acts
 torch.cuda.synchronize()
 return dict(sum_scores=sums,token_logprobs=tokvals,token_counts=counts,probe_nll=probe_loss,probe=probe_record,geometry=geom,update_projections=projection,controls=stats,elapsed_s=time.monotonic()-begin)
def load():
 assert importlib.metadata.version('lm_eval')==CFG['lm_eval_version']
 random.seed(2021);np.random.seed(2021);torch.manual_seed(2021);torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 directory=BASE/'models/Ouro-1.4B';meta=json.loads((directory/'snapshot_metadata.json').read_text());assert meta['sha']==CFG['model_revision']
 tok=AutoTokenizer.from_pretrained(directory,local_files_only=True)
 model,info=AutoModelForCausalLM.from_pretrained(directory,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
 assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
 model=model.to('cuda').eval();model.requires_grad_(False);assert model.model.total_ut_steps==4
 assert all(p.dtype==torch.bfloat16 and not p.requires_grad for p in model.parameters())
 dump('results/model_load.json',dict(revision=meta['sha'],loading=info,torch=torch.__version__,harness=importlib.metadata.version('lm_eval'),frozen=True,dtype='bfloat16',config=model.config.to_dict()))
 return tok,model
def parity(tok,model,row):
 from lm_eval.models.huggingface import HFLM
 from lm_eval.api.instance import Instance
 model.early_exit_step=3
 wrapper=HFLM(pretrained=model,tokenizer=tok,backend='causal',batch_size=1,device='cuda',max_length=8192,add_bos_token=False,dtype='bfloat16',trust_remote_code=True)
 scored=score_view(tok,model,row['variants']['official'],row['question'],True)
 req=[Instance(request_type='loglikelihood',doc={},arguments=tuple(p),idx=i,metadata=(row['task'],i,1)) for i,p in enumerate(row['variants']['official'])]
 expected=wrapper.loglikelihood(req,disable_tqdm=True)
 errors=[abs(a[0]-b)/n for a,b,n in zip(expected,scored['sum_scores']['keep'],scored['token_counts'])]
 assert max(errors)<=.05,errors
 assert np.argmax(np.array([x[0] for x in expected])/row['choice_char_lengths'])==np.argmax(np.array(scored['sum_scores']['keep'])/row['choice_char_lengths'])
 x=tok('The Earth orbits the Sun.',return_tensors='pt').to('cuda');_,states,_=model.model(**x,use_cache=False)
 native=model(**x,use_cache=False,exit_at_step=3,logits_to_keep=1).logits
 assert torch.equal(native,model.lm_head(states[-1][:,-1:,:]))
 return dict(passed=True,harness_error_per_token=errors,native_final_exact=True,causal_probe=scored['probe'])
def baseline_timing(tok,model,row):
 torch.cuda.synchronize();begin=time.monotonic();sums=[]
 for prompt,answer in row['variants']['official']:
  prefix,target=encode(tok,prompt,answer);x=torch.tensor([prefix+target[:-1]],device='cuda')
  _,states,_=model.model(input_ids=x,use_cache=False)
  z1=model.lm_head(states[0][:,len(prefix)-1:,:])[0].float();z4=model.lm_head(states[3][:,len(prefix)-1:,:])[0].float()
  top=z4.softmax(-1).topk(2,dim=-1).values;w=1-top[:,0]+top[:,1];z=z4+w[:,None]*(z4-z1)
  sums.append(float(z.log_softmax(-1).gather(-1,torch.tensor(target,device='cuda')[:,None]).sum()))
 torch.cuda.synchronize();return dict(seconds=time.monotonic()-begin,sum_scores=sums)
def main():
 split=sys.argv[1];pilot=split=='pilot';source='discovery' if pilot else split
 audit=json.loads((ROOT/'data/manifest_audit.json').read_text());path=ROOT/'data'/(source+'.jsonl')
 assert sha(path)==audit['manifests'][source]['sha256'];rows=read('data/'+source+'.jsonl')
 if pilot:rows=rows[:CFG['pilot_questions']]
 if source.startswith('final'):assert (ROOT/'configs/FINAL_LOCK.json').exists(),'Final policy must be locked'
 signature={p.name:sha(p) for p in [Path(__file__),ROOT/'configs/protocol.json',path]}
 out=ROOT/'results'/(split+'.jsonl');assert not out.exists(),'Never overwrite previous predictions; audit explicit resume separately'
 tok,model=load();timings=[]
 if pilot:dump('results/PARITY.json',parity(tok,model,rows[0]))
 torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
 with out.open('x',encoding='utf8') as f:
  for i,row in enumerate(rows):
   if time.time()+30>float(os.environ['WORKER_DEADLINE']):raise RuntimeError('Budget reserve')
   if __import__('shutil').disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Disk reserve')
   # Scorer never receives gold or candidate correctness.
   views={k:score_view(tok,model,pairs,row['question']) for k,pairs in row['variants'].items()}
   if pilot:
    timing=baseline_timing(tok,model,row);errors=[abs(a-b)/n for a,b,n in zip(timing['sum_scores'],views['official']['sum_scores']['adaptive'],views['official']['token_counts'])]
    assert max(errors)<.05
    assert np.argmax(np.array(timing['sum_scores'])/row['choice_char_lengths'])==np.argmax(np.array(views['official']['sum_scores']['adaptive'])/row['choice_char_lengths'])
    timings.append(dict(id=row['id'],baseline=timing['seconds'],all_readouts_and_probes=views['official']['elapsed_s'],two_views=sum(v['elapsed_s'] for v in views.values()),max_per_token_error=max(errors)))
   result=dict(id=row['id'],task=row['task'],gold=row['gold'],choice_char_lengths=row['choice_char_lengths'],views=views,signature=signature)
   f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
   dump('STATUS.json',dict(stage=split,completed=i+1,expected=len(rows),elapsed_s=time.monotonic()-begin,peak_vram_bytes=torch.cuda.max_memory_allocated(),pid=os.getpid()))
   if (i+1)%10==0:print(json.dumps(dict(stage=split,completed=i+1,expected=len(rows))),flush=True)
 dump('results/'+split+'_COMPLETE.json',dict(passed=True,n=len(rows),sha256=sha(out),elapsed_s=time.monotonic()-begin,peak_vram_bytes=torch.cuda.max_memory_allocated(),signature=signature))
 if pilot:dump('results/PILOT_COST.json',dict(n=len(rows),timings=timings,baseline_seconds=sum(t['baseline'] for t in timings),all_readouts_probe_seconds=sum(t['all_readouts_and_probes'] for t in timings),two_views_seconds=sum(t['two_views'] for t in timings)))
if __name__=='__main__':main()
