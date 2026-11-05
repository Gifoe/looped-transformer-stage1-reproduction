"""CPU-only saved ABI/task/resource preflight, no tokenizer/model construction."""
import json
from r30_common import ROOT,read,sha,digest
from r30_cache_abi import validate
from r30_protocol import build
def run():
 abi=validate();graph,tasks,uses,plan=build();assert plan==read('configs/r30_CLOCK_TASK_PLAN.json');old=read('configs/r28_SOURCE_PRIOR_PILOT_LOCK.json')
 for p,h in old['source_sha256'].items():assert sha(p)==h
 ledgers=dict(old['budget']['inherited_ledger_sha256']);ledgers['logs/r28_gpu_budget.json']=sha('logs/r28_gpu_budget.json');assert len(ledgers)==17 and sum(read(p)['used_seconds'] for p in ledgers)==20194.085665407125
 nativekeys=set(plan['native_tasks']);assert all(tasks[k]['mode']=='native' for k in nativekeys)
 for g in plan['groups']:
  for name,k in g['reference_CF'].items():
   t=tasks[k];n=tasks[g['reference_native'][name]];assert t['input_ids_sha256']==n['input_ids_sha256'] and t['length']==n['length'] and t['context']==name and t['native_task']==n['key'] and t['key']!=n['key']
 return dict(status='passed_CPU_only_preflight',ABI=abi,plan_sha256=sha('configs/r30_CLOCK_TASK_PLAN.json'),source_positions=143,groups=389,CF_tasks=778,model_calls=785,native_head_passes=2,native_heads=7194,body_heads=787,total_heads=7981,inherited_ledger_sha256=ledgers,inherited_total_seconds=20194.085665407125,remaining_seconds=1405.914334592875,model_loaded=False,tokenizer_constructed=False,GPU_worker_seconds=0,scope='Saved ABI/input graph only, no real native heads replayed and no clock features computed')
if __name__=='__main__':
 result=run()
 with (ROOT/'results/r30_CPU_PREFLIGHT.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['ABI','inherited_ledger_sha256']}))
