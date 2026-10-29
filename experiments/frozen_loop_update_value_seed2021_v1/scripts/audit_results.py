"""Independent arithmetic, chronology, ID and accounting checks on completed runs."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def rows(p):
 with (ROOT/p).open(encoding='utf8') as f:return [json.loads(l) for l in f if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 manifest=load('data/manifest_audit.json');lock=load('configs/FINAL_LOCK.json');pre=load('configs/PREVALIDATION_LOCK.json');checks={};seen=set()
 for name in ['discovery','head_train','validation','final_sciq','final_piqa']:
  source=rows('data/'+name+'.jsonl');actual=rows('results/'+name+'.jsonl');complete=load('results/'+name+'_COMPLETE.json');receipt=load('logs/'+name+'_exit.json')
  assert len(source)==len(actual)==manifest['manifests'][name]['n']==complete['n']
  assert sha(ROOT/'data'/(name+'.jsonl'))==manifest['manifests'][name]['sha256']
  assert sha(ROOT/'results'/(name+'.jsonl'))==complete['sha256']
  assert receipt['exit_code']==0
  counts={a:0 for a in ['adaptive','keep','forward','rollback','depth1','depth2','negative']};missing=0
  for s,r in zip(source,actual):
   assert s['id']==r['id'] and s['gold']==r['gold'] and s['choice_char_lengths']==r['choice_char_lengths']
   assert r['id'] not in seen;seen.add(r['id'])
   assert r['signature']['protocol.json']==pre['files']['configs/protocol.json']
   for v in r['views'].values():
    assert len(v['update_projections'])==96 and np.isfinite(v['update_projections']).all()
    assert v['probe']['positions']==[t-1 for t in v['probe']['token_indices']]
    assert all(0<t<v['probe']['prefix_tokens'] for t in v['probe']['token_indices'])
    for a,scores in v['sum_scores'].items():
     assert np.isfinite(scores).all()
     assert np.allclose(scores,[sum(x) for x in v['token_logprobs'][a]],atol=5e-4,rtol=1e-6)
    assert all((x is None) if not v['probe']['token_indices'] else (x is not None and np.isfinite(x)) for x in v['probe_nll'].values())
   v=r['views']['official'];missing+=int(v['probe_nll']['keep'] is None)
   for a in counts:counts[a]+=int(np.argmax(np.array(v['sum_scores'][a])/r['choice_char_lengths'])==r['gold'])
  if name=='validation':assert pre['frozen_unix']<receipt['started_unix']<lock['frozen_unix']
  if name.startswith('final'):assert lock['frozen_unix']<receipt['started_unix']
  checks[name]=dict(n=len(actual),fixed_action_correct=counts,official_probe_missing_questions=missing,raw_sha256=complete['sha256'])
 for rel,h in pre['files'].items():assert sha(ROOT/rel)==h,('Prevalidation rule changed',rel)
 for rel,h in lock['frozen_files'].items():assert sha(ROOT/rel)==h
 budget=load('logs/round_budget.json');assert abs(budget['used_seconds']-sum(r['seconds'] for r in budget['runs']))<1e-5
 assert budget['used_seconds']<=28800 and budget['historical_start_seconds']+budget['used_seconds']<=180000
 analysis=load('results/FINAL_ANALYSIS.json')
 for split in ['final_sciq','final_piqa']:
  for a,n in checks[split]['fixed_action_correct'].items():assert n==analysis['reports'][split]['summary'][a]['correct']
 report=dict(passed=True,total_disjoint_questions=len(seen),datasets=checks,chronology_verified=True,source_locks_verified=True,budget_seconds=budget['used_seconds'],cumulative_seconds=budget['historical_start_seconds']+budget['used_seconds'])
 (ROOT/'results/INDEPENDENT_FINAL_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
if __name__=='__main__':main()
