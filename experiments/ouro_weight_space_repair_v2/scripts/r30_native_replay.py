"""Frozen-native ABI reader and label-free head/scalar parity gate.

Only normalized single-position BF16 states are read. FullV tensors live for
one shared-reference group; no old spatial CF is a clock-CF cache hit.
"""
import numpy as np
from r28_common import ROOT,read,digest,PositionBuffers
from r28_stream import task_ids
from r7_math import token_logits
BASE_METHODS=['J2_guidance','M2','M3','Fixed','native1','native4','native5','adaptive','N0_native1','N0_native4','N0_Fixed']
class NativeCache:
 def __init__(self,device):
  self.index=read('results/r28_NATIVE_STATE_INDEX.json');self.records={r['task']:r for r in self.index['tasks']};self.arrays=np.load(ROOT/'results/r28_NATIVE_SINGLE_POSITION_STATES.npz',allow_pickle=False);self.device=device;self.head_calls=0
 def close(self):self.arrays.close()
 def states(self,task):
  import torch
  r=self.records[task['key']];assert task['mode']=='native' and r['input_ids_sha256']==task['input_ids_sha256'] and r['context']==task['context'] and r['depth']==task['depth']
  result=[]
  for ref in r['states']:
   a=self.arrays[ref['key']];assert a.dtype==np.uint16 and a.shape==(1,1,2048);s=torch.from_numpy(a.copy()).view(torch.bfloat16).to(self.device);result.append(s)
  return result
 def heads(self,task,model,measure):
  import torch
  with measure('cache_IO_decode_transfer/'+task['context']):states=self.states(task)
  heads={}
  for k in ([1,4,5] if task['context']=='Q' else [1,4]):
   h=states[k-1];assert h.shape==(1,1,2048) and h.dtype==torch.bfloat16
   with measure('head/cache/'+task['context']):
    raw=model.lm_head(h);assert raw.dtype==torch.bfloat16 and raw.shape==(1,1,49152);heads[k]=raw.float()[0,0];self.head_calls+=1
   assert torch.isfinite(heads[k]).all()
  return states,heads
def baseline_logits(q,n1,n4):
 import torch
 q1,q4=q[1],q[4];top=q4.reshape(1,-1).softmax(-1).topk(2,dim=-1).values[0]
 return dict(J2_guidance=token_logits(q1,q4,n1[1:],n4[1:])[0],M2=token_logits(q1,q4,n1[:1],n4[:1])[0],M3=token_logits(q1,q4,n1,n4)[0],Fixed=q4+.5*(q4-q1),native1=q1,native4=q4,native5=q[5],adaptive=q4+(1-(top[0]-top[1]))*(q4-q1))
def fill_native_group(group,tasks,cache,model,measure,uses,choices,buffers):
 import torch
 refs={}
 for name in ['N0','N1','N2']:
  task=tasks[group['reference_tasks'][name+'_native']];_,refs[name]=cache.heads(task,model,measure)
 with measure('projection/native_baseline_setup'):
  n1=torch.stack([refs[n][1] for n in ['N0','N1','N2']]);n4=torch.stack([refs[n][4] for n in ['N0','N1','N2']]);prior={n:z.reshape(1,-1).log_softmax(-1)[0] for n,z in dict(N0_native1=refs['N0'][1],N0_native4=refs['N0'][4],N0_Fixed=refs['N0'][4]+.5*(refs['N0'][4]-refs['N0'][1])).items()}
 for key in group['Q_consumers']:
  _,q=cache.heads(tasks[key],model,measure)
  with measure('projection/native_baseline_normalization_buffers'):
   logs={n:z.reshape(1,-1).log_softmax(-1)[0] for n,z in baseline_logits(q,n1,n4).items()}
   for u in uses[key]:
    c=choices[(u['id'],u['candidate'])];t=u['Q_position']
    for name,z in logs.items():buffers.put(u['id'],u['candidate'],name,t,z[c['Q_target_ids'][t]])
    for name,z in prior.items():buffers.put(u['id'],u['candidate'],name,t,z[c['reference_target_ids']['N0'][t]])
def verify_scalar_parity(rows,finished,old):
 """No gold/argmax: strict equality of every baseline LP/sum/score/late/R6."""
 from slcd_r6_scores import selected_score
 oldrows={r['id']:r for r in old['rows']};count=0;lp_count=0
 for row in rows:
  prior={n:dict(sum_scores=[],normalized_scores=[]) for n in ['native_k1','native_k4','loopcd_k4_fixed']};original={n:dict(sum_scores=[],normalized_scores=[]) for n in ['native_k1','native_k4','native_k5','loopcd_k4_fixed','loopcd_k4_adaptive']};orow=oldrows[row['id']];assert row['split']==orow['split'] and len(row['candidates'])==len(orow['candidates'])
  for j,c in enumerate(row['candidates']):
   o=orow['candidates'][j];chars=c['characters'];assert chars>0 and o['candidate']==j and o['characters']==chars and o['tokens']==len(c['Q_target_ids'])
   pieces={n:finished[(row['id'],j,n)] for n in BASE_METHODS}
   for name,p in pieces.items():assert p==o['pieces'][name],(row['id'],j,name);lp_count+=len(p['target_log_probabilities'])
   late=pieces['native4']['sum_score']/chars-pieces['native5']['sum_score']/chars;assert late==o['late']
   for name,p in pieces.items():
    if name.startswith('N0_'):continue
    score=p['sum_score']/chars+(late if name not in ['native1','native4','native5','Fixed','adaptive'] else 0.);assert score==o['scores'][name]
   assert pieces['Fixed']['sum_score']/chars+late==o['scores']['A1']
   for n,k in [('native_k1','native1'),('native_k4','native4'),('native_k5','native5'),('loopcd_k4_fixed','Fixed'),('loopcd_k4_adaptive','adaptive')]:original[n]['sum_scores'].append(pieces[k]['sum_score']);original[n]['normalized_scores'].append(pieces[k]['sum_score']/chars)
   for n,k in [('native_k1','N0_native1'),('native_k4','N0_native4'),('loopcd_k4_fixed','N0_Fixed')]:prior[n]['sum_scores'].append(pieces[k]['sum_score']);prior[n]['normalized_scores'].append(pieces[k]['sum_score']/chars)
   count+=1
  assert original==orow['matched_original_methods'] and prior==orow['matched_N0_prior_methods']
  for c,v in zip(orow['candidates'],selected_score(original,prior,'r6_prior_projection')):assert float(v)==c['scores']['R6_matched']
 assert count==96
 return dict(status='passed_strictzero_native_baseline_scalar_parity',candidates=count,per_target_LPs=lp_count,late_exact=True,matched_R6_exact=True,gold_read=False,argmax_used=False,scope='All baseline scalar fields; historical fullV exact is checked only for nine saved representative heads')
