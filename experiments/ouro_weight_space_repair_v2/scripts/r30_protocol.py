"""New clock-CF identities over unchanged known-input native graph."""
from r28_common import read,digest
from r28_stream import stream_contract,task_ids,source_contract
def build():
 graph,old,uses,checks=stream_contract();positions,_,_=source_contract();assert len(positions)==143
 native={k:t for k,t in old.items() if t['mode']=='native'};tasks=dict(native);groups=[]
 for g in graph['stream_groups']:
  refs={n:g['reference_tasks'][n+'_native'] for n in ['N0','N1','N2']}
  cf={}
  for n in ['N1','N2']:
   t=dict(native[refs[n]]);ids=task_ids(graph,t);assert all(0<p<len(ids)-1 for p in positions)
   identity=dict(input_ids=ids,context=n,mode='source_clock_loop1_to_loop4_same_layer_same_positions',depth=4,source_positions=list(positions),capture_loop_index=0,patch_loop_index=3,primitive='original_uncached_sdpa_BF16_single_position_head')
   key=digest(identity);assert key not in old and key not in tasks;t.update(key=key,mode='clock_CF',clock_identity_sha256=key,native_task=refs[n],source_positions_sha256=digest(list(positions)));tasks[key]=t;cf[n]=key
  groups.append(dict(reference_native=refs,reference_CF=cf,Q_consumers=list(g['Q_consumers'])))
 reps={n:next(k for k in checks['no_hook_native_tasks'] if native[k]['context']==n) for n in ['N1','N2']}
 assert len(native)==1588 and len(tasks)==2366 and len(groups)==389 and len({k for g in groups for k in g['reference_CF'].values()})==778
 plan=dict(status='CPU_clock_task_definition',native_tasks=list(native),clock_CF_tasks=[k for k in tasks if k not in native],groups=groups,representative_native=reps,fresh_Q_task=checks['exact_duplicate_Q_task'],source_positions=list(positions),source_positions_sha256=digest(list(positions)),base_CF_forwards=778,extra_integrity=dict(fresh_Q=1,capture_only=2,identity=2,CF_repeat=2),maximum_body_forwards=785,native_heads_per_pass=3597,native_head_passes=2,base_CF_heads=778,scope='Old spatial-CF task keys are not reused; no model, gold, scores or argmax in construction')
 return graph,tasks,uses,plan
if __name__=='__main__':
 from pathlib import Path
 import json
 root=Path(__file__).resolve().parents[1];_,_,_,plan=build()
 with (root/'configs/r30_CLOCK_TASK_PLAN.json').open('x',encoding='utf8') as f:json.dump(plan,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in plan.items() if k not in ['groups','native_tasks','clock_CF_tasks','source_positions']}))
