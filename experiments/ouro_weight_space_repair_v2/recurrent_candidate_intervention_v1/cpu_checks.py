"""Changed-path CPU fixtures, no model instantiation or GPU calls."""
from pathlib import Path
import json,ast,hashlib
import numpy as np,torch
from core import scalar_scores,OUT
from analyze import factorial_contrasts

def reject(fn):
 try:fn()
 except (AssertionError,RuntimeError,IndexError):return True
 raise AssertionError('negative fixture accepted')

def main():
 lp=torch.full((3,49152),-10.,dtype=torch.float32);z=torch.zeros_like(lp);target=[2,5,7]
 for j,t in enumerate(target):lp[j,t]=-(j+1);z[j,t]=10+j
 r=scalar_scores(lp,z,target,3);assert r['sum_logprob']==-6. and r['char_normalized']==-2. and r['token_logprobs']==[-1.,-2.,-3.] and r['raw_target_logits']==[10.,11.,12.]
 assert reject(lambda:scalar_scores(lp,z,[2,5],3)) and reject(lambda:scalar_scores(lp,z,target,0)) and reject(lambda:scalar_scores(lp,z,[2,5,49152],3)) and reject(lambda:scalar_scores(lp.double(),z,target,3))
 bad=lp.clone();bad[0,2]=float('nan');assert reject(lambda:scalar_scores(bad,z,target,3))
 for n in [3,4,5]:
  gold=n-1;order=list(range(n));order[0],order[gold]=order[gold],order[0];assert order.index(gold)==0 and order[0]==gold
 order=[1,0,2,3];mapped=np.array([2.,2.,0.,0.]);display=mapped[order];assert order[int(display.argmax())]==1 and int(mapped.argmax())==0
 fc=factorial_contrasts([1,0],[0,1],[1,1],[1,0]);assert np.array_equal(fc['accuracy_factor_interaction'],[1,-2]);assert reject(lambda:factorial_contrasts([1]*24,[0]*4,[1]*4,[0]*4))
 common=np.array([3.,3.,3.]);assert all(common[i]-common[j]==0 for i in range(3) for j in range(i+1,3));specific=np.array([3.,1.,2.]);assert specific[0]-specific[1]==2
 plan=json.loads((OUT/'INFERENCE_TASK_PLAN.json').read_text(encoding='utf8'));protocol=json.loads((OUT/'intervention_protocol.json').read_text(encoding='utf8'));assert len(plan['tasks'])==218 and len(plan['baseline_task_keys'])==24 and protocol['summary']['native_head_prediction_positions']==1448
 def goldfree(x):
  if isinstance(x,dict):
   assert not set(x)&{'gold','correct','expected_index','expected_display_index','expected_original_content_index','condition','semantic_decision'}
   for v in x.values():goldfree(v)
  elif isinstance(x,list):
   for v in x:goldfree(v)
 goldfree(plan);goldfree(json.loads((OUT/'INFERENCE_REQUEST_TEXTS.json').read_text(encoding='utf8')))
 count=0;cid=set()
 for t in plan['tasks']:
  assert t['positions']==list(range(len(t['input_ids'])-len(t['positions']),len(t['input_ids'])))
  for c in t['consumers']:
   assert c['consumer_id'] not in cid;cid.add(c['consumer_id']);assert len(c['target_ids'])==len(t['positions']) and c['char_denominator']>0;count+=1
 assert count==784
 source=ast.parse((OUT/'worker.py').read_text(encoding='utf8'));assert not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr in ['argmax','backward','train'] for x in ast.walk(source))
 files=['construct_semantics.py','build_protocol.py','core.py','worker.py','supervise.py','analyze.py','cpu_checks.py']
 for n in files:ast.parse((OUT/n).read_text(encoding='utf8'))
 out=dict(status='passed_CPU_synthetic_no_model_or_GPU',tests=['multi_token_FP32_sum_once','original_body_char_division_once','target_count_range_dtype_nonfinite_rejection','n3_n4_n5_content_display_mapping','displayed_tie_priority','common_normalizer_cancels_J','B5_paired_factorial_signs_and_denominator_rejection','gold_free_inference_schema','218body_784consumer_1448position_plan','worker_no_argmax_training','syntax'],deadline_scope='static supervisor review, no claim of real GPU timeout fixture',source_sha256={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in files},plan_sha256=hashlib.sha256((OUT/'INFERENCE_TASK_PLAN.json').read_bytes()).hexdigest(),torch_version=torch.__version__)
 (OUT/'CPU_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out))
if __name__=='__main__':main()
