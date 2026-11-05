"""Complete immutable task stream and original-position LL reduction."""
from r28_common import read,digest
METHODS=['primary','J2_guidance','M2','M3','G_matched','vocabulary_null','LOO_N1','LOO_N2','Fixed','matched_scale','native1','native4','native5','adaptive','N0_native1','N0_native4','N0_Fixed']
def stream_contract():
 graph=read('configs/r28_TRUE_PREFIX_TASK_INVENTORY.json');assert graph['status']=='passed_CPU_prefix_admission';tasks={t['key']:t for t in graph['tasks']};uses={};observed=[]
 for u in graph['usages']:uses.setdefault(u['tasks']['Q'],[]).append(u)
 for g in graph['stream_groups']:
  assert set(g['reference_tasks'])=={'N0_native','N1_native','N1_CF','N2_native','N2_CF'} and g['Q_consumers']
  observed.extend(g['reference_tasks'].values());observed.extend(g['Q_consumers'])
 assert len(observed)==len(set(observed))==len(tasks)==2366 and set(observed)==set(tasks)
 for r in graph['candidate_records']:
  assert r['reference_position_maps']['N0']==list(range(r['reference_target_positions']['N0'])) and r['reference_target_positions']['N0']==r['Q_target_positions']
 duplicate=next(t['key'] for t in graph['tasks'] if t['context']=='Q' and len(uses[t['key']])>1)
 checks=dict(no_hook_native_tasks=[next(t['key'] for t in graph['tasks'] if t['context']==n and t['mode']=='native') for n in ['N1','N2']],identity_native_tasks=[next(t['key'] for t in graph['tasks'] if t['context']==n and t['mode']=='native') for n in ['N1','N2']],exact_duplicate_Q_task=duplicate,base_model_forwards=2366,extra_integrity_forwards=5,total_forwards_if_complete=2371)
 return graph,tasks,uses,checks

def task_ids(graph,task):
 ids=graph['prefix_pool'][task['prefix_ref']]['ids']+task['consumed_ids'];assert digest(ids)==task['input_ids_sha256'] and len(ids)==task['length'];return ids

def source_contract():
 blocks=read('results/r28_ANSWER_BLOCK_INVENTORY.json');lookup={b['demo_index']:b for b in blocks['blocks']};mapping=[]
 for m in blocks['cycle_mapping']:
  a=lookup[m['recipient_demo_index']]['positions'];b=lookup[m['donor_demo_index']]['positions'];assert len(a)==len(b);mapping.extend(zip(a,b))
 mapping=sorted(mapping);positions=tuple(p for p,_ in mapping);index={p:i for i,p in enumerate(positions)};permutation=tuple(index[q] for _,q in mapping);assert len(positions)==143 and sorted(permutation)==list(range(143));return positions,permutation,digest(mapping)

def candidate_readouts(rows,finished):
 import numpy as np
 from slcd_r6_scores import selected_score
 result=[]
 for row in rows:
  candidates=[];original={n:dict(sum_scores=[],normalized_scores=[]) for n in ['native_k1','native_k4','native_k5','loopcd_k4_fixed','loopcd_k4_adaptive']};prior={n:dict(sum_scores=[],normalized_scores=[]) for n in ['native_k1','native_k4','loopcd_k4_fixed']}
  for j,c in enumerate(row['candidates']):
   pieces={name:finished[(row['id'],j,name)] for name in METHODS};chars=c['characters'];assert chars>0
   late=pieces['native4']['sum_score']/chars-pieces['native5']['sum_score']/chars
   scores={name:v['sum_score']/chars+(late if name not in ['native1','native4','native5','Fixed','adaptive'] else 0.) for name,v in pieces.items() if not name.startswith('N0_')};scores['A1']=pieces['Fixed']['sum_score']/chars+late
   for name,source in [('native_k1','native1'),('native_k4','native4'),('native_k5','native5'),('loopcd_k4_fixed','Fixed'),('loopcd_k4_adaptive','adaptive')]:original[name]['sum_scores'].append(pieces[source]['sum_score']);original[name]['normalized_scores'].append(pieces[source]['sum_score']/chars)
   for name,source in [('native_k1','N0_native1'),('native_k4','N0_native4'),('loopcd_k4_fixed','N0_Fixed')]:prior[name]['sum_scores'].append(pieces[source]['sum_score']);prior[name]['normalized_scores'].append(pieces[source]['sum_score']/chars)
   candidates.append(dict(candidate=j,characters=chars,tokens=len(c['Q_target_ids']),late=late,pieces=pieces,scores=scores))
  r6=selected_score(original,prior,'r6_prior_projection')
  for c,v in zip(candidates,r6):c['scores']['R6_matched']=float(v)
  result.append(dict(id=row['id'],split=row['split'],candidates=candidates,matched_original_methods=original,matched_N0_prior_methods=prior))
 return result
