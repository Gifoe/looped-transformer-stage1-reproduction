"""Real new-math outputs -> 19-piece scalar analyzer, entirely synthetic."""
import copy,json,hashlib,ast
from pathlib import Path
import numpy as np,torch
from r33_math import torch_readouts
from r33_common import pack_diag,OUTPUTS,ROOT
from r33_stream import METHODS,candidate_readouts
from r33_analyze import scalar_integrity
def reject(f):
 try:f()
 except AssertionError:return
 raise AssertionError('Negative scalar fixture accepted')
def finish(values):
 out={}
 for key,vs in values.items():
  a=torch.tensor(vs,dtype=torch.float32);out[key]=dict(sum_score=float(a.sum()),target_log_probabilities=a.tolist())
 return out
def main():
 rng=np.random.default_rng(2021);q1=torch.from_numpy(rng.normal(size=37).astype('float32'));q4=torch.from_numpy(rng.normal(size=37).astype('float32'));n1=torch.from_numpy(rng.normal(size=(3,37)).astype('float32'));n4=torch.from_numpy(rng.normal(size=(3,37)).astype('float32'));perm=torch.from_numpy(rng.permutation(37));packets={}
 for zero in [False,True]:
  out,d=torch_readouts(q1,q4,n4 if zero else n1,n4,perm);assert bool(d['h_exact_zero'])==zero;top=q4.softmax(-1).topk(2).values;out.update(native1=q1,native4=q4,native5=q4+.25*q1,adaptive=q4+(1-(top[0]-top[1]))*(q4-q1));out.update(N0_native1=(n4 if zero else n1)[0],N0_native4=n4[0],N0_Fixed=n4[0]+.5*(n4[0]-(n4 if zero else n1)[0]));assert set(out)==set(METHODS);packets[zero]=({k:v.log_softmax(-1) for k,v in out.items()},pack_diag(d))
 rows=[];tasks={};uses={};diags={};values={};count=0
 for i in range(24):
  row=dict(id='synthetic_'+str(i),split='synthetic',candidates=[])
  for j in range(4):
   size=6 if i*4+j<68 else 5;target=[(j+t)%37 for t in range(size)];c=dict(candidate=j,characters=3+j,Q_target_ids=target,reference_target_ids={'N0':target});row['candidates'].append(c)
   for name in METHODS:values[(row['id'],j,name)]=[]
   for t,token in enumerate(target):
    key='synthetic_prefix_'+str(count%421);index=count%421;count+=1;zero=index%3==0;logs,d=packets[zero];tasks[key]=dict(context='Q',mode='native');diags[key]=copy.deepcopy(d);uses.setdefault(key,[]).append(dict(id=row['id'],candidate=j,Q_position=t))
    for name,z in logs.items():values[(row['id'],j,name)].append(float(z[token]))
  rows.append(row)
 assert count==548 and len(tasks)==421
 finished=finish(values);scores=dict(rows=candidate_readouts(rows,finished),diagnostics=diags);rebuilt,unmatched=scalar_integrity(scores,rows,tasks,uses,vocab_size=37);assert rebuilt==finished and not unmatched
 bad=copy.deepcopy(scores);bad['rows'][0]['candidates'][0]['pieces'].pop('LOO_N0');reject(lambda:scalar_integrity(bad,rows,tasks,uses,37))
 bad=copy.deepcopy(scores);key=next(iter(diags));bad['diagnostics'][key]['alpha']=0.;bad['diagnostics'][key].pop('main_norm2');reject(lambda:scalar_integrity(bad,rows,tasks,uses,37))
 active=next(k for k,d in diags.items() if not d['h_exact_zero']);bad=copy.deepcopy(scores);bad['diagnostics'][active]['main_norm2']=0.;reject(lambda:scalar_integrity(bad,rows,tasks,uses,37))
 bad=copy.deepcopy(scores);bad['diagnostics'][active]['consensus_norm_error']=1e-5;reject(lambda:scalar_integrity(bad,rows,tasks,uses,37))
 # Valid degenerate consensus remains a recorded qualification failure.
 changed=copy.deepcopy(values)
 for u in uses[active]:changed[(u['id'],u['candidate'],'reference_consensus_matched')][u['Q_position']]=changed[(u['id'],u['candidate'],'Fixed')][u['Q_position']]
 bad=copy.deepcopy(scores);bad['rows']=candidate_readouts(rows,finish(changed));d=bad['diagnostics'][active];d.update(consensus_unmatched=True,consensus_raw_norm2=0.,consensus_threshold=0.,consensus_norm_error=float(np.sqrt(d['main_norm2'])));_,unmatched=scalar_integrity(bad,rows,tasks,uses,37);assert unmatched and unmatched[0]['control']=='consensus'
 # Recompute the altered sum/score so only exact-zero routing can detect it.
 zero=next(k for k,d in diags.items() if d['h_exact_zero']);u=uses[zero][0];changed=copy.deepcopy(values);changed[(u['id'],u['candidate'],'primary')][u['Q_position']]=float(np.float32(changed[(u['id'],u['candidate'],'primary')][u['Q_position']]-.25));bad=copy.deepcopy(scores);bad['rows']=candidate_readouts(rows,finish(changed));reject(lambda:scalar_integrity(bad,rows,tasks,uses,37))
 assert all(not(ROOT/p).exists() for p in OUTPUTS);source=(ROOT/'scripts/r33_analyze.py').read_text(encoding='utf8');ast.parse(source);main=source[source.index('def main():'):];assert main.index('numeric_integrity(spec)')<main.index('if unmatched:')<main.index('result=analyze_effects(')
 result=dict(status='passed_CPU_synthetic_19_method_scalar_pipeline',questions=24,candidates=96,position_uses=548,synthetic_prefixes=421,pieces=19,old_alpha_schema_rejected=True,missing_LOO_N0_rejected=True,main_zero_norm_equivalence_enforced=True,matched_consensus_norm_error_rejected=True,valid_consensus_unmatched_recorded_no_effect=True,exact_zero_primary_mismatch_rejected=True,CPU_FP32_sum_scope='Schema/rounding-bound fixture only, not claimed bit-exact GPU reduction',model_loaded=False,new_head_run=False,GPU_worker_seconds=0,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/r33_stream.py','scripts/r33_analyze.py','scripts/r33_common.py','scripts/r33_cpu_stream_checks.py']})
 with (ROOT/'results/r33_CPU_STREAM_CHECKS.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
