"""Frozen-plan native-loop collector. No new decoder, no gold/argmax in worker."""
from pathlib import Path
import sys,json,hashlib,time,os,inspect
ROOT=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
H=lambda b:hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def packed_diag(d):
 import torch
 return {k:v.detach().cpu().item() if isinstance(v,torch.Tensor) else v for k,v in d.items()}
def target_sum(z,target):
 import torch
 assert z.ndim==2 and len(target)==z.shape[0] and z.shape[1]==49152 and all(isinstance(t,int) and 0<=t<49152 for t in target)
 t=torch.tensor(target,device=z.device,dtype=torch.long);p=z.log_softmax(-1).gather(-1,t[:,None]).squeeze(-1);assert bool(torch.isfinite(p).all());return float(p.sum())
def all_heads(model,states):
 import torch
 assert states and all(s.dtype==torch.bfloat16 and s.ndim==3 and s.shape[0]==1 and s.shape[-1]==2048 for s in states)
 out=[]
 for s in states:
  z=model.lm_head(s.to(model.lm_head.weight.device));assert z.dtype==torch.bfloat16 and z.shape[:2]==s.shape[:2] and z.shape[-1]==49152
  z=z[0].float();assert bool(torch.isfinite(z).all());out.append(z)
 return out

def main():
 started=time.monotonic();lock=read(OUT/'a4_EXECUTION_LOCK.json');assert lock['status']=='frozen_native_loop_audit'
 for rel,h in lock['source_sha256'].items():assert H((OUT/rel).read_bytes())==h
 planpath=OUT/'a4_CPU_TASK_PLAN.json';assert H(planpath.read_bytes())==lock['plan_sha256'];plan=read(planpath)
 for rel,h in plan['source_inputs'].items():assert H((ROOT/rel).read_bytes())==h
 # Source/model/inputs admitted before load. No gold is present in this task plan.
 assert all('gold' not in r for k in ['parity','cohort64'] for r in plan[k])
 for name,h in plan['model_identity']['metadata_and_tokenizer_sha256'].items():assert H((ROOT/'models/Ouro-1.4B'/name).read_bytes())==h
 assert H((ROOT/'models/Ouro-1.4B/model.safetensors').read_bytes())==plan['model_identity']['safetensors_sha256']
 from runtime import load_model,check_budget
 from ouro_cached_scoring import OuroCachedScorer
 from weight_edits import tensor_sha256
 import numpy as np,torch
 tok,model=load_model();scorer=OuroCachedScorer(model,4)
 assert H(Path(inspect.getfile(type(model))).read_bytes())=='c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413'
 def params():return {n:tensor_sha256(v) for n,v in model.named_parameters()}
 before=params();assert len(before)==269;baseline_hooks={n:(len(m._forward_hooks),len(m._forward_pre_hooks)) for n,m in model.named_modules()};arrays={};index={};times=[];bodycalls=0;headcalls=0;paritychecks=[]
 def vector(a,dtype):
  b=a.detach().cpu().contiguous();raw=b.view(torch.uint16).numpy().copy() if dtype=='BF16' else b.numpy().copy();key=H((dtype+str(raw.shape)).encode()+raw.tobytes());arrays.setdefault(key,raw);index[key]=dict(dtype=dtype,shape=list(raw.shape));return key
 def run(task,depth,purpose):
  nonlocal bodycalls,headcalls
  check_budget(5);scorer.kmax=depth;scorer._set_depth(depth);torch.cuda.synchronize();t=time.monotonic()
  _,s,_=scorer._run(task['input_ids'],positions=task['positions'],use_cache=False);assert len(s)==depth
  z=all_heads(model,s);bodycalls+=1;headcalls+=depth;torch.cuda.synchronize();elapsed=time.monotonic()-t
  times.append(dict(purpose=purpose,depth=depth,length=len(task['input_ids']),seconds=elapsed))
  return s,z,elapsed
 def lps(z,target):
  return target_sum(z,target)
 def question(row,guard):
  output=dict(id=row['id'],rotations=[])
  for rot in row['rotations']:
   memo={};keys={}
   for j,task in enumerate(rot['tasks']):
    s,z,elapsed=run(task,4,'parity_native4' if guard else 'cohort_native4');memo[j]=z;keys[j]=[vector(x,'BF16') for x in s]
    if guard:
     for depth in [2,3]:
      ss,zz,_=run(task,depth,'independent_depth'+str(depth));a=vector(s[depth-1],'BF16');b=vector(ss[-1],'BF16');c=vector(z[depth-1],'FP32');d=vector(zz[-1],'FP32');assert a==b and c==d,('independent native parity',row['id'],rot['shift'],depth)
      for pair in rot['pairs']:
       if pair['task_index']==j:assert lps(z[depth-1],pair['target_ids'])==lps(zz[-1],pair['target_ids'])
      paritychecks.append(dict(id=row['id'],shift=rot['shift'],task=j,depth=depth,state_key=a,independent_state_key=b,head_key=c,independent_head_key=d,exact=True))
   mapped={str(k):[None]*row['nchoices'] for k in [1,2,3,4]};fixed=[None]*row['nchoices'];targetlp=[]
   for pos,pair in enumerate(rot['pairs']):
    z=memo[pair['task_index']];original=rot['order'][pos];vals=[lps(x,pair['target_ids']) for x in z]
    for k,v in enumerate(vals,1):mapped[str(k)][original]=v
    f=z[3]+.5*(z[3]-z[0]);fixed[original]=lps(f,pair['target_ids']);targetlp.append(vals)
   assert mapped['1']==rot['expected_K1'] and mapped['4']==rot['expected_K4'] and fixed==rot['expected_Fixed'],('historical endpoint parity',row['id'],rot['shift'])
   output['rotations'].append(dict(shift=rot['shift'],order=rot['order'],native_sum_scores=mapped,fixed_sum_scores=fixed,state_keys=keys,target_lp_sums_displayed=targetlp,task_keys=[t['key'] for t in rot['tasks']]))
  return output
 parity=[question(r,True) for r in plan['parity']];save(OUT/'a4_PARITY.json',dict(status='all_exact',questions=parity,independent_checks=paritychecks))
 # Runtime-only admission: no outcome-dependent choice; 2x max observed per-length^2 rate plus fixed finalization reserve.
 native=[x for x in times if x['purpose']=='parity_native4'];rate=max(x['seconds']/x['length']**2 for x in native);tasks=[t for r in plan['cohort64'] for rot in r['rotations'] for t in rot['tasks']];estimate=2*rate*sum(len(t['input_ids'])**2 for t in tasks)+60.;remaining=float(os.environ['GPU_DEADLINE_UNIX'])-time.time();admitted=estimate+5<remaining
 save(OUT/'a4_RUNTIME_ADMISSION.json',dict(admitted=admitted,forecast_seconds_including_60s_finalization=estimate,remaining_to_soft_deadline=remaining,observed_native4= native,rule='2 * max observed native4 time / length^2 * sum planned length^2 +60 finalization; require another5sec reserve; measured scheduling may change, hard supervisor still applies'))
 completed=[]
 if admitted:
  with (OUT/'a4_RAW.jsonl').open('x',encoding='utf8') as f:
   for r in plan['cohort64']:
    q=question(r,False);f.write(json.dumps(q,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno());completed.append(r['id']);save(OUT/'a4_PROGRESS.json',dict(completed=len(completed),target=64,body_calls=bodycalls,head_calls=headcalls))
  assert len(completed)==64 and bodycalls==plan['counts']['total_model_calls'] and headcalls==1092
 after=params();assert before==after and baseline_hooks=={n:(len(m._forward_hooks),len(m._forward_pre_hooks)) for n,m in model.named_modules()}
 np.savez(OUT/'a4_NATIVE_VECTORS.npz',**arrays);save(OUT/'a4_VECTOR_INDEX.json',index);save(OUT/'a4_TIMINGS.json',times)
 save(OUT/'a4_COMPLETE.json',dict(status='complete64' if admitted else 'parity_only_runtime_not_admitted',completed_ids=completed,body_calls=bodycalls,head_calls=headcalls,parameter_hashes_before=before,parameter_hashes_after=after,cleanup_exact=True,vector_npz_sha256=H((OUT/'a4_NATIVE_VECTORS.npz').read_bytes()),raw_sha256=H((OUT/'a4_RAW.jsonl').read_bytes()) if admitted else None,plan_sha256=H(planpath.read_bytes()),protocol_sha256=H((OUT/'a4_EXECUTION_LOCK.json').read_bytes()),worker_seconds=time.monotonic()-started,numerical_effects_unsealed=False))
 print('A4_TERMINAL', 'complete64' if admitted else 'parity_only_runtime_not_admitted',bodycalls,flush=True)
if __name__=='__main__':main()
