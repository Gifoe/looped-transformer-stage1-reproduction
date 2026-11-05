"""Shared immutable input gates, scalar helpers and atomic output writes."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(name,x):
 p=OUT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
OUTPUTS=['SCORES.jsonl','COMPLETE.json','INPUT_AUDIT.json','PARITY.json','RUNTIME_ADMISSION.json','VECTORS.npz','VECTOR_INDEX.json','TIMINGS.json','PROGRESS.json','SUPERVISOR.json','gpu_budget.json','worker.out','worker.err','ANALYSIS.json','numerical_validation.json','permutation_transitions.csv','distractor_effects.csv','content_letter_readout.csv','question_level_statistics.csv','native_predictions.csv','native_loop_transitions.csv']
def gate():
 s=read(OUT/'EXECUTION_LOCK.json');assert s['status']=='frozen_bounded_input_intervention'
 for n,h in s['source_sha256'].items():assert H((OUT/n).read_bytes())==h,n
 for n,h in s['input_sha256'].items():assert H((ROOT/n).read_bytes())==h,n
 for n,h in s['inherited_ledgers'].items():assert H((ROOT/n).read_bytes())==h,n
 return s

def scalar_scores(lp,z,target,chars):
 import torch
 assert lp.dtype==z.dtype==torch.float32 and lp.ndim==2 and lp.shape==z.shape and lp.shape[1]==49152 and len(target)==lp.shape[0] and isinstance(chars,int) and chars>0
 assert all(isinstance(t,int) and 0<=t<49152 for t in target)
 t=torch.tensor(target,device=lp.device,dtype=torch.long);v=lp.gather(-1,t[:,None]).squeeze(-1);raw=z.gather(-1,t[:,None]).squeeze(-1);assert bool(torch.isfinite(v).all()) and bool((v<=0).all());total=float(v.sum())
 return dict(sum_logprob=total,char_normalized=total/chars,char_denominator=chars,token_logprobs=v.detach().cpu().tolist(),raw_target_logits=raw.detach().cpu().tolist())
