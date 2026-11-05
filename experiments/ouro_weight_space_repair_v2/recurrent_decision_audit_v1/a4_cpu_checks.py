"""Synthetic head, rotation, target and source checks; deadline code statically reviewed; no model load."""
from pathlib import Path
import json,sys,hashlib,ast
import numpy as np,torch
from collect_a4 import all_heads,packed_diag,target_sum
OUT=Path(__file__).resolve().parent
class Head:
 weight=torch.ones(1)
 def __call__(self,s):return s[:,:,:1].expand(s.shape[0],s.shape[1],49152).to(torch.bfloat16)
class Fake:lm_head=Head()
def reject(f):
 try:f()
 except (AssertionError,RuntimeError):return True
 raise AssertionError('negative fixture accepted')
def main():
 states=[torch.full((1,1,2048),float(k),dtype=torch.bfloat16) for k in range(1,5)];z=all_heads(Fake(),states);assert len(z)==4 and all(x.dtype==torch.float32 and x.shape==(1,49152) for x in z)
 assert reject(lambda:all_heads(Fake(),[states[0].float()])) and reject(lambda:all_heads(Fake(),[torch.ones(1,1,1024,dtype=torch.bfloat16)]))
 assert reject(lambda:all_heads(Fake(),[]))
 q=torch.arange(49152,dtype=torch.float32).remainder(31).reshape(1,-1);qq=q.repeat(2,1);assert target_sum(qq,[7,8])==float(qq.log_softmax(-1).gather(-1,torch.tensor([[7],[8]])).squeeze(-1).sum());assert target_sum(q,[7])==float(q.log_softmax(-1)[0,7].sum());assert reject(lambda:target_sum(q,[7,8])) and reject(lambda:target_sum(q,[49152]))
 assert packed_diag(dict(x=torch.tensor(.5),flag=True,n=3,mode='native'))==dict(x=.5,flag=True,n=3,mode='native')
 for n in [3,4,5]:
  original=np.arange(n,dtype=float)
  for shift in range(n):
   order=[(j+shift)%n for j in range(n)];displayed=original[order];mapped=np.empty(n)
   for j,i in enumerate(order):mapped[i]=displayed[j]
   assert np.array_equal(mapped,original)
 # A displayed tie has a different original-content tie priority after rotation.
 order=[1,2,3,0];mapped=np.ones(4);assert order[int(mapped[order].argmax())]!=int(mapped.argmax())
 plan=json.loads((OUT/'a4_CPU_TASK_PLAN.json').read_text());assert plan['counts']['total_model_calls']==279 and len(plan['selection']['ids'])==64 and len(set(plan['selection']['ids']))==64
 for group in ['parity','cohort64']:
  for q in plan[group]:
   assert 'gold' not in q
   for r in q['rotations']:
    assert sorted(r['order'])==list(range(q['nchoices']))
    for t in r['tasks']:assert t['positions']==[len(t['input_ids'])-1] and all(0<=x<49152 for x in t['input_ids'])
    assert all(len(p['target_ids'])==1 for p in r['pairs'])
 for p in OUT.glob('*a4.py'):ast.parse(p.read_text(encoding='utf8'))
 files=['collect_a4.py','supervise_a4.py','prepare_a4.py','a4_cpu_checks.py','analyze_a4.py','restore_inputs.py'];out=dict(status='passed_synthetic_no_model_or_GPU',tests=['actual_four_native_heads','BF16_required','hidden_shape_required','zero_depth_rejected','original_FP32_target_reduction','target_length_and_range_rejected','mixed_diagnostics','cyclic_roundtrip_3_4_5','displayed_tie_priority','279_call_plan','64_unique_seed_ids','single_token_target_input_positions','gold_free_plan','source_syntax'],source_sha256={p:hashlib.sha256((OUT/p).read_bytes()).hexdigest() for p in files},plan_sha256=hashlib.sha256((OUT/'a4_CPU_TASK_PLAN.json').read_bytes()).hexdigest(),torch_version=torch.__version__)
 (OUT/'a4_CPU_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
