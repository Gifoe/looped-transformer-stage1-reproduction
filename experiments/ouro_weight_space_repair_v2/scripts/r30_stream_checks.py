"""Synthetic scalar parity and task identity/order guards; no real head or label."""
import json,hashlib,copy,ast
from pathlib import Path
import numpy as np,torch
from r30_protocol import build
from r30_native_replay import BASE_METHODS,baseline_logits,verify_scalar_parity
from r28_stream import METHODS,candidate_readouts
from r28_math_v2 import torch_readouts,numpy_readouts
ROOT=Path(__file__).resolve().parents[1]
def rejected(f):
 try:f()
 except AssertionError:return
 raise AssertionError('Invalid fixture accepted')
def main():
 from r30_worker import pack_diag
 mixed={'tensor':torch.tensor(1.),'bool':True,'int':3,'str':'native'};assert pack_diag(mixed)=={'tensor':1.,'bool':True,'int':3,'str':'native'}
 rng=np.random.default_rng(2021);maxerr=0.
 for _ in range(8):
  q={k:torch.from_numpy(rng.normal(size=37).astype('float32')) for k in [1,4,5]};n1=torch.from_numpy(rng.normal(size=(3,37)).astype('float32'));n4=torch.from_numpy(rng.normal(size=(3,37)).astype('float32'));b4=torch.from_numpy(rng.normal(size=(2,37)).astype('float32'));perm=torch.from_numpy(rng.permutation(37));base=baseline_logits(q,n1,n4);out,diag=torch_readouts(q[1],q[4],n1,n4,b4,perm);packed=pack_diag(diag);json.dumps(packed,allow_nan=False);npout,_=numpy_readouts(q[1].numpy(),q[4].numpy(),n1.numpy(),n4.numpy(),b4.numpy(),perm.numpy())
  for name in ['J2_guidance','M2','M3','Fixed']:assert torch.equal(base[name],out[name])
  for name in out:maxerr=max(maxerr,float(np.abs(out[name].numpy()-npout[name]).max()))
 assert maxerr<=1e-6
 rows=[];finished={}
 for i in range(24):
  row=dict(id='synthetic_'+str(i),split='synthetic',candidates=[])
  for j in range(4):
   row['candidates'].append(dict(candidate=j,characters=3+j,Q_target_ids=[1,2,3],reference_target_ids={'N0':[1,2,3]}))
   for name in METHODS:
    v=torch.tensor([-1.-j,-2.-i/32,-3.],dtype=torch.float32);finished[(row['id'],j,name)]=dict(sum_score=float(v.sum()),target_log_probabilities=v.tolist())
  rows.append(row)
 old=dict(rows=candidate_readouts(rows,finished));basefinished={k:v for k,v in finished.items() if k[2] in BASE_METHODS};receipt=verify_scalar_parity(rows,basefinished,old);assert receipt['candidates']==96
 bad=copy.deepcopy(basefinished);bad[('synthetic_0',0,'native1')]['target_log_probabilities'][0]=float(np.nextafter(np.float32(-1),np.float32(0)));rejected(lambda:verify_scalar_parity(rows,bad,old))
 bad=copy.deepcopy(old);bad['rows'][0]['candidates'][0]['scores']['R6_matched']+=1e-9;rejected(lambda:verify_scalar_parity(rows,basefinished,bad))
 graph,tasks,uses,plan=build();oldkeys={t['key'] for t in graph['tasks']};assert set(plan['clock_CF_tasks']).isdisjoint(oldkeys) and plan['maximum_body_forwards']==785 and len(plan['groups'])==389
 worker=(ROOT/'scripts/r30_worker.py').read_text(encoding='utf8');ast.parse(worker);assert worker.index('abi=validate()')<worker.index('tok,model=load_model()') and worker.index('parity=verify_scalar_parity(')<worker.index("qtask=tasks[plan['fresh_Q_task']]")<worker.index('with SourceClock(model)')
 analyzer=(ROOT/'scripts/r30_analyze.py').read_text(encoding='utf8');ast.parse(analyzer);main=analyzer[analyzer.index('def main():'):];assert main.index('numeric_integrity(spec)')<main.index('if unmatched:')<main.index('result=analyze_effects(')
 result=dict(status='passed_CPU_synthetic_stream',FP32_numpy_torch_max_error=maxerr,synthetic_candidates=96,one_ULP_target_LP_rejected=True,R6_change_rejected=True,new_CF_identity_disjoint=True,native_tasks=1588,CF_tasks=778,model_calls_if_complete=785,native_heads=7194,body_heads=787,total_heads_if_complete=7981,model_loaded=False,GPU_worker_seconds=0,source_sha256={p.as_posix():hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [Path('scripts/r30_native_replay.py'),Path('scripts/r30_protocol.py'),Path('scripts/r30_worker.py'),Path('scripts/r30_analyze.py'),Path('scripts/r30_stream_checks.py')]})
 with (ROOT/'results/r30_CPU_STREAM_CHECKS.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
