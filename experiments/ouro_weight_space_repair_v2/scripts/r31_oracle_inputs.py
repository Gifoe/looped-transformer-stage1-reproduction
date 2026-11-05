"""Join all fixed 548 uses by actual Q prefix; no model/state/head access."""
from r31_oracle_common import read,sha,digest
from r31_oracle_math import exact,coefficients
def load():
 complete=read('results/r30_COMPLETE.json');scores=read('results/r30_SCORES.json');analysis=read('results/r30_ANALYSIS.json');graph=read('configs/r28_TRUE_PREFIX_TASK_INVENTORY.json');requests=read('configs/r28_SANITIZED_REQUESTS.json')
 assert complete['status']=='complete' and complete['n']==24 and complete['scores_sha256']==sha('results/r30_SCORES.json') and analysis['status']=='complete_analyzed_once' and analysis['scores_sha256']==complete['scores_sha256'] and analysis['protocol_sha256']==complete['protocol_sha256']==sha('configs/r30_SOURCE_CLOCK_PILOT_LOCK.json')
 tasks={t['key']:t for t in graph['tasks']};score_rows={r['id']:r for r in scores['rows']};pred={r['id']:r for r in analysis['predictions']};request_rows=requests['rows'];assert len(request_rows)==len(score_rows)==len(pred)==24
 maps={};owners={};used=0
 for u in graph['usages']:
  identity=(u['id'],u['candidate'],u['Q_position']);assert identity not in maps;k=u['tasks']['Q'];t=tasks[k];assert t['context']=='Q' and t['mode']=='native' and t['read_position']==t['length']-1
  ids=graph['prefix_pool'][t['prefix_ref']]['ids']+t['consumed_ids'];assert len(ids)==t['length'] and digest(ids)==t['input_ids_sha256'];maps[identity]=k;owners.setdefault(k,set()).add(u['id'])
 result=[]
 for row in request_rows:
  qid=row['id'];out=score_rows[qid];labels=pred[qid];assert len(row['candidates'])==len(out['candidates'])==4 and row['split']==out['split'];chains=[];pieces=[];chars=[];late=[]
  for j,(candidate,rec) in enumerate(zip(row['candidates'],out['candidates'])):
   assert candidate['candidate']==rec['candidate']==j and rec['tokens']==len(candidate['Q_target_ids']) and candidate['characters']==rec['characters']>0
   f=rec['pieces']['Fixed']['target_log_probabilities'];p=rec['pieces']['primary']['target_log_probabilities'];assert len(f)==len(p)==len(candidate['Q_target_ids'])>0
   import numpy as np
   for v in f+p:assert float(np.float32(v))==v and np.isfinite(v) and v<=0
   chain=[maps[(qid,j,t)] for t in range(len(f))];chains.append(chain);used+=len(chain);pieces.append((f,p));chars.append(candidate['characters']);late.append(rec['late'])
   for t,k in enumerate(chain):
    task=tasks[k];assert task['consumed_ids']==candidate['Q_target_ids'][:t]
  problem=coefficients(pieces,chains,chars,late);assert len(problem['keys'])+len(problem['zero_keys'])==len(set(k for ch in chains for k in ch))
  for j,rec in enumerate(out['candidates']):
   # Endpoints enclose actual stored GPU FP32 sum / FP64 final score.
   zero=problem['a'][j];one=zero+sum(problem['d'][j],exact(0));assert abs(zero-exact(rec['scores']['A1']))<=problem['B'][j] and abs(one-exact(rec['scores']['primary']))<=problem['B'][j]
  y=labels['gold'];assert type(y)==int and 0<=y<4;old=labels['predictions'];background=dict(R6_matched_correct=old['R6_matched']==y,R6_historical_correct=old['R6_historical']==y,R6_matched_only_gain=old['R6_matched']==y and old['Fixed']!=y,R6_historical_only_gain=old['R6_historical']==y and old['Fixed_historical']!=y)
  result.append(dict(id=qid,split=row['split'],gold=y,problem=problem,chains=chains,characters=chars,late=late,background=background))
 assert used==len(maps)==548 and len(owners)==421 and sum(len(r['chains']) for r in result)==96
 cross={k:sorted(v) for k,v in owners.items() if len(v)>1}
 return result,dict(rows=24,candidates=96,position_uses=548,unique_Q_prefix_keys=421,cross_question_shared_keys=cross,independent_question_witnesses_jointly_compatible=not cross,source_sha256=sha('configs/r28_TRUE_PREFIX_TASK_INVENTORY.json'),scope='If any cross-question key is shared, only single-question capacity upper bounds are claimed and joint reachability is undetermined')
