"""R30 complete label-free integrity precedes inherited single effect analysis."""
import math,hashlib
import numpy as np
from r30_common import ROOT,LOCK,read,sha,digest,save,gate
from r30_protocol import build
from r30_native_replay import BASE_METHODS,verify_scalar_parity
from r28_stream import METHODS,candidate_readouts
from r28_analyze import analyze_effects

def vectors(path,obj):
 refs={}
 def scan(x):
  if isinstance(x,dict):
   if set(x)=={'key','dtype','shape','raw_sha256'}:
    if x['key'] in refs:assert refs[x['key']]==x
    refs[x['key']]=x
   else:
    for v in x.values():scan(v)
  elif isinstance(x,list):
   for v in x:scan(v)
 scan(obj)
 with np.load(ROOT/path,allow_pickle=False) as z:
  assert set(z.files)==set(refs)
  for key,r in refs.items():
   a=z[key];assert r['dtype'] in ['torch.bfloat16','torch.float32'] and list(a.shape)==r['shape'] and a.dtype==np.dtype('uint16' if r['dtype']=='torch.bfloat16' else 'float32');b=a.tobytes();assert hashlib.sha256(b).hexdigest()==r['raw_sha256'] and hashlib.sha256((r['dtype']+str(tuple(r['shape']))).encode()+b).hexdigest()==key
   values=(a.astype(np.uint32)<<16).view(np.float32) if r['dtype']=='torch.bfloat16' else a;assert np.isfinite(values).all()
 return refs

def numeric_integrity(spec):
 c=read('results/r30_COMPLETE.json');sup=read('logs/r30_SUPERVISOR.json');assert sup['status']=='complete' and sup['exit_code']==0 and not sup.get('timed_out',False) and not sup['overrun'] and 0<sup['gpu_worker_seconds']<=700
 old=spec['budget']['inherited_ledger_sha256'];assert len(old)==17
 for p,h in old.items():assert sha(p)==h
 inherited=sum(read(p)['used_seconds'] for p in old);assert inherited==20194.085665407125;book=read('logs/r30_gpu_budget.json');assert book['used_seconds']==sup['gpu_worker_seconds'] and book['inherited_total_seconds']==inherited and book['total_used_seconds']==inherited+book['used_seconds']<=21600
 assert c['status']=='complete' and c['n']==24 and c['body_forwards']==785 and c['CF_forwards']==778 and c['groups']==389 and c['fatal'] is None and c['protocol_sha256']==sha(LOCK) and c['native_head_calls']==7194
 assert len(c['all_parameter_hashes_before'])==269 and c['all_parameter_hashes_before']==c['all_parameter_hashes_after']==read('results/r28_COMPLETE.json')['all_parameter_hashes_after']
 for field,p in [('scores_sha256','results/r30_SCORES.json'),('audit_sha256','results/r30_EXECUTION_AUDIT.json'),('representative_vectors_sha256','results/r30_REPRESENTATIVE_VECTORS.npz'),('native_parity_sha256','results/r30_NATIVE_PARITY.json')]:assert c[field]==sha(p)
 assert all(math.isfinite(v) and v>=0 for v in c['component_seconds'].values()) and abs(sum(c['component_seconds'].values())-c['measured_components_sum'])<=1e-8 and c['measured_components_sum']<=c['elapsed_seconds']+1e-6<=sup['gpu_worker_seconds']+1e-6
 audit=read('results/r30_EXECUTION_AUDIT.json');cleanup=c['cleanup'];assert cleanup==audit['cleanup'] and cleanup['handles_removed'] and cleanup['original_hooks_restored'] and cleanup['all_model_hooks_empty'] and not cleanup['errors'] and audit['fatal'] is None and audit['native_head_calls']==7194
 parity=read('results/r30_NATIVE_PARITY.json');assert parity['status']=='passed_strictzero_native_baseline_scalar_parity' and parity['candidates']==96 and parity['native_head_calls']==3597 and parity['historical_saved_fullV_heads_checked']==9 and not parity['gold_read'] and not parity['argmax_used']
 graph,tasks,uses,plan=build();calls=audit['calls'];assert len(calls)==785 and [x['number'] for x in calls]==list(range(1,786))
 base=[x for x in calls if x['purpose']=='base'];assert len(base)==len({x['task'] for x in base})==778 and {x['task'] for x in base}==set(plan['clock_CF_tasks'])==set(audit['CF_tasks'])
 for purpose,expected in [('fresh_Q',[plan['fresh_Q_task']]),('native_capture_only',list(plan['representative_native'].values())),('identity',list(plan['representative_native'].values())),('repeat',[next(x['key'] for x in tasks.values() if x.get('native_task')==k) for k in plan['representative_native'].values()])]:assert sorted(x['task'] for x in calls if x['purpose']==purpose)==sorted(expected)
 for r in calls:
  t=tasks[r['task']];assert all(r[k]==t[k] for k in ['context','depth','input_ids_sha256','length']) and r['head_input_shape']==[1,1,2048]
  if r['purpose']=='fresh_Q':assert r['mode']=='no_hook' and r['clock'] is None;continue
  mode={'base':'clock_CF','native_capture_only':'capture_only','identity':'identity','repeat':'clock_CF'}[r['purpose']];clock=r['clock'];assert r['mode']==clock['mode']==mode and clock['context']==t['context'] and clock['input_ids_sha256']==t['input_ids_sha256'] and clock['source_positions']==plan['source_positions'] and clock['calls_per_layer']==[4]*24 and clock['events']==[[loop,i] for loop in range(4) for i in range(24)] and clock['capture_loop_index']==0 and clock['spatial_map']=='identity' and clock['donor_layer']=='same_physical_layer'
  assert clock['patch_loop_index']==(None if mode=='capture_only' else 3) and clock['applied_donor_loop_index']==(None if mode=='capture_only' else 3 if mode=='identity' else 0)
  if mode!='capture_only':assert clock['other_rows_raw_exact']
  if r['purpose']=='base':assert r['first3_last_position_exact'] and len(r['states'])==4 and all(s['dtype']=='torch.bfloat16' and s['shape']==[1,1,2048] for s in r['states'])
 types=[x['type'] for x in audit['checks']];assert types.count('fresh_Q_exact')==types.count('first_real_math')==1 and all(types.count(k)==2 for k in ['capture_only_native_exact','CF_first3_full_and_loop1_source_exact','identity_exact','CF_repeat_exact'])
 refs=vectors('results/r30_REPRESENTATIVE_VECTORS.npz',audit)
 with np.load(ROOT/'results/r30_REPRESENTATIVE_VECTORS.npz',allow_pickle=False) as z,np.load(ROOT/'results/r28_NATIVE_SINGLE_POSITION_STATES.npz',allow_pickle=False) as native:
  index={x['task']:x for x in read('results/r28_NATIVE_STATE_INDEX.json')['tasks']}
  for r in base:
   for a,b in zip(r['states'][:3],index[tasks[r['task']]['native_task']]['states'][:3]):assert z[a['key']].tobytes()==native[b['key']].tobytes()
  for x in audit['checks']:
   typ=x['type']
   if typ=='fresh_Q_exact':assert x['exact'] and x['fresh']==x['cached']
   elif typ=='capture_only_native_exact':assert x['exact'] and len(x['donors'])==24 and all(r['shape']==[1,143,spec['projection_width']] for r in x['donors'])
   elif typ=='CF_first3_full_and_loop1_source_exact':assert x['exact'] and x['native']['states'][:3]==x['CF']['states'][:3] and len(x['donors'])==24
   elif typ=='identity_exact':assert x['exact'] and x['native']['states']==x['identity']['states'] and x['native']['heads']['4']==x['identity']['heads']['4']
   elif typ=='CF_repeat_exact':
    assert x['exact'] and x['first']==x['repeat'];first=next(a for a in audit['checks'] if a['type']=='CF_first3_full_and_loop1_source_exact' and a['task']==x['task']);assert first['donors']==x['donors']
   elif typ=='first_real_math':assert x['max_FP32_output_error']<=1e-6 and x['max_FP64_diag_error']<=1e-10
 scores,rows,unmatched=scalar_integrity(tasks);finished={(r['id'],c['candidate'],name):p for r in scores['rows'] for c in r['candidates'] for name,p in c['pieces'].items() if name in BASE_METHODS};verify_scalar_parity(rows,finished,read('results/r28_SCORES.json'))
 return c,scores,rows,unmatched

def scalar_integrity(tasks):
 scores=read('results/r30_SCORES.json');rows=read('configs/r28_SANITIZED_REQUESTS.json')['rows'];assert len(scores['rows'])==len(rows)==24;finished={}
 for out,row in zip(scores['rows'],rows):
  assert out['id']==row['id'] and out['split']==row['split'] and len(out['candidates'])==len(row['candidates'])
  for rec,ch in zip(out['candidates'],row['candidates']):
   assert rec['candidate']==ch['candidate'] and rec['characters']==ch['characters'] and rec['tokens']==len(ch['Q_target_ids']) and set(rec['pieces'])==set(METHODS)
   for name,piece in rec['pieces'].items():
    values=np.asarray(piece['target_log_probabilities'],dtype=np.float32);size=len(ch['reference_target_ids']['N0']) if name.startswith('N0_') else len(ch['Q_target_ids']);assert values.shape==(size,) and np.isfinite(values).all() and np.all(values<=0) and math.isfinite(piece['sum_score']) and float(np.float32(piece['sum_score']))==piece['sum_score']
    # Predeclared IEEE FP32 reduction error certificate; observed GPU one-sum value remains authoritative.
    exact=math.fsum(float(v) for v in values);scale=math.fsum(abs(float(v)) for v in values);u=np.finfo(np.float32).eps;gamma=(max(size-1,0)*u)/(1-max(size-1,0)*u);assert abs(piece['sum_score']-exact)<=gamma*scale+np.finfo(np.float64).eps*scale
    finished[(row['id'],ch['candidate'],name)]=piece
 assert candidate_readouts(rows,finished)==scores['rows']
 assert set(scores['diagnostics'])=={k for k,t in tasks.items() if t['context']=='Q' and t['mode']=='native'};unmatched=[]
 for key,d in scores['diagnostics'].items():
  for name,value in d.items():
   if isinstance(value,(int,float)):assert math.isfinite(value),name
  assert 0<=d['alpha']<=1 and d['denominator']>=0 and d['indistinguishable_threshold']>=0 and d['scale_g2']>=0 and d['scale_r2']>=0 and d['scale_rho']>=0
  if d['denominator']<=d['indistinguishable_threshold'] or d['numerator']<=0:assert d['alpha']==0
  else:assert abs(d['alpha']-min(1.,d['numerator']/d['denominator']))<=64*np.finfo(np.float64).eps
  for tag in ['G','null']:
   if d[tag+'_unmatched']:unmatched.append(dict(task=key,control=tag,raw_norm2=d[tag+'_raw_norm2'],norm_error=d[tag+'_norm_error']))
   else:assert d[tag+'_norm_error']<=1e-10
 return scores,rows,unmatched

def main():
 assert not (ROOT/'results/r30_ANALYSIS.json').exists(),'Official analyzer is single-use'
 spec=gate();c,scores,rows,unmatched=numeric_integrity(spec)
 save('results/r30_LABEL_FREE_INTEGRITY.json',dict(status='passed_complete_label_free_numeric',controls_matched=not unmatched,unmatched=unmatched,protocol_sha256=sha(LOCK),scores_sha256=c['scores_sha256'],gold_or_argmax_accessed=False))
 if unmatched:save('results/r30_ANALYSIS.json',dict(status='complete_but_unmatched_no_effect_unsealing',unmatched=unmatched,strong_screen=False,cost_review_eligible=False,protocol_sha256=sha(LOCK)));return
 result=analyze_effects(scores,rows);result.update(status='complete_analyzed_once',protocol_sha256=sha(LOCK),scores_sha256=c['scores_sha256']);save('results/r30_ANALYSIS.json',result)
if __name__=='__main__':main()
