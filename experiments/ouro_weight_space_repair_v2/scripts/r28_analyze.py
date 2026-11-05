"""Complete numerical/source/accounting integrity before the only gold/argmax pass."""
import math,json,hashlib
import numpy as np
from r28_common import ROOT,LOCK,read,sha,digest,save
from r28_stream import stream_contract,task_ids,source_contract,METHODS,candidate_readouts

def numeric_integrity(spec):
 from r28_common import gate
 gate();c=read('results/r28_COMPLETE.json');sup=read('logs/r28_SUPERVISOR.json');assert sup['status']=='complete' and sup['exit_code']==0 and not sup.get('timed_out',False) and not sup['overrun'] and 0<sup['gpu_worker_seconds']<=spec['budget']['hard_seconds']
 old=spec['budget']['inherited_ledger_sha256'];assert len(old)==16
 for p,h in old.items():assert sha(p)==h
 inherited=sum(read(p)['used_seconds'] for p in old);assert inherited==18930.428655151278;book=read('logs/r28_gpu_budget.json');assert book['used_seconds']==sup['gpu_worker_seconds'] and book['inherited_total_seconds']==inherited and book['total_used_seconds']==inherited+book['used_seconds']<=21600
 assert c['status']=='complete' and c['n']==24 and c['forwards']==2371 and c['base_tasks']==2366 and c['groups']==389 and c['fatal'] is None and c['protocol_sha256']==sha(LOCK)
 assert len(c['all_parameter_hashes_before'])==269 and c['all_parameter_hashes_before']==c['all_parameter_hashes_after']==read('results/r27_EXEC_COMPLETE.json')['all_parameter_hashes_after']
 for field,p in [('scores_sha256','results/r28_SCORES.json'),('audit_sha256','results/r28_EXECUTION_AUDIT.json'),('representative_vectors_sha256','results/r28_REPRESENTATIVE_VECTORS.npz'),('native_states_sha256','results/r28_NATIVE_SINGLE_POSITION_STATES.npz'),('native_state_index_sha256','results/r28_NATIVE_STATE_INDEX.json')]:assert c[field]==sha(p)
 assert all(math.isfinite(v) and v>=0 for v in c['component_seconds'].values());assert abs(sum(c['component_seconds'].values())-c['measured_components_sum'])<=1e-8 and c['measured_components_sum']<=c['elapsed_seconds']+1e-6<=sup['gpu_worker_seconds']+1e-6
 audit=read('results/r28_EXECUTION_AUDIT.json');cleanup=c['cleanup'];assert cleanup==audit['cleanup'] and cleanup['handles_removed'] and cleanup['original_hooks_restored'] and cleanup['all_model_hooks_empty'] and not cleanup['errors'] and audit['fatal'] is None
 graph,tasks,uses,checks=stream_contract();positions,permutation,mapsha=source_contract();assert mapsha==spec['source_map_sha256']==audit['source_map_sha256'];calls=audit['calls'];assert len(calls)==2371 and [r['number'] for r in calls]==list(range(1,2372))
 base=[r for r in calls if r['purpose']=='base'];assert len(base)==len({r['task'] for r in base})==2366 and {r['task'] for r in base}==set(tasks)
 extras=[r for r in calls if r['purpose']!='base'];assert sorted(r['task'] for r in extras if r['purpose']=='no_hook_native')==sorted(checks['no_hook_native_tasks']) and sorted(r['task'] for r in extras if r['purpose']=='identity_replay')==sorted(checks['identity_native_tasks']) and [r['task'] for r in extras if r['purpose']=='exact_duplicate']==[checks['exact_duplicate_Q_task']]
 cf_count=0
 for r in calls:
  t=tasks[r['task']];assert all(r[k]==t[k] for k in ['context','depth','input_ids_sha256','length']) and r['head_input_shape']==[1,1,2048]
  if r['purpose']=='no_hook_native':assert r['clock'] is None;continue
  clock=r['clock'];mode='patch' if r['purpose']=='identity_replay' or t['mode']=='answer_block_CF4' else 'capture' if t['context'] in ['N1','N2'] else 'off';assert r['mode']==clock['mode']==mode and clock['context']==t['context'] and clock['input_ids_sha256']==t['input_ids_sha256'] and clock['calls_per_layer']==[t['depth']]*24 and clock['events_sha256']==digest([[loop,i] for loop in range(t['depth']) for i in range(24)])
  if mode=='off':assert clock['source_positions']==[]
  else:
   assert clock['source_positions']==list(positions) and clock['permutation']==(list(range(143)) if r['purpose']=='identity_replay' else list(permutation))
   if mode=='patch':assert clock['other_rows_raw_exact'] and clock['native_donor_only']
  if r['purpose']=='base' and t['mode']=='answer_block_CF4':
   cf_count+=1;assert r['first3_full_sequence_exact'] and r['donor_native_input_sha256']==t['input_ids_sha256'] and r['source_map_sha256']==mapsha;nt=tasks[r['native_task']];assert nt['mode']=='native' and nt['context']==t['context'] and nt['input_ids_sha256']==t['input_ids_sha256']
 assert cf_count==778
 types=[r['type'] for r in audit['checks']];assert types.count('no_hook_capture_exact')==types.count('identity_exact')==types.count('first3_full_representative')==2 and types.count('first_actual_duplicate_exact')==types.count('first_real_math')==types.count('N0_representative')==1
 for r in audit['checks']:
  if r['type']=='no_hook_capture_exact':assert r['exact'] and r['no_hook']==r['native'] and len(r['donors'])==24 and all(d['shape']==[1,143,spec['projection_width']] for d in r['donors'])
  elif r['type']=='identity_exact':assert r['exact'] and r['native']['states']==r['identity']['states'] and r['native']['heads']['4']==r['identity']['heads']['4']
  elif r['type']=='first3_full_representative':assert r['exact'] and r['native']==r['CF'] and len(r['native'])==3
  elif r['type']=='first_actual_duplicate_exact':assert r['exact'] and r['A']==r['B']
  elif r['type']=='first_real_math':assert r['task'] in tasks and tasks[r['task']]['context']=='Q' and r['max_FP32_output_error']<=1e-6 and r['max_FP64_diag_error']<=1e-10
 def vector_audit(path,obj):
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
  return len(refs)
 vector_audit('results/r28_REPRESENTATIVE_VECTORS.npz',audit['checks']);index=read('results/r28_NATIVE_STATE_INDEX.json');assert index['future_reuse_requires_separate_review'] and index['weights_sha256']==spec['model_identity']['safetensors_sha256'] and index['parameter_hash_digest']==digest(c['all_parameter_hashes_before']) and index['cache_contract_sha256']==spec['cache_contract_sha256']
 natives=[r for r in index['tasks']];expected={k for k,t in tasks.items() if t['mode']=='native'};assert len(natives)==len({r['task'] for r in natives})==1588 and {r['task'] for r in natives}==expected and sum(len(r['states']) for r in natives)==6773
 for r in natives:assert len(r['states'])==tasks[r['task']]['depth'] and r['input_ids_sha256']==tasks[r['task']]['input_ids_sha256'] and all(v['dtype']=='torch.bfloat16' and v['shape']==[1,1,2048] for v in r['states'])
 vector_audit('results/r28_NATIVE_SINGLE_POSITION_STATES.npz',index)
 scores=read('results/r28_SCORES.json');rows=read('configs/r28_SANITIZED_REQUESTS.json')['rows'];assert len(scores['rows'])==len(rows)==24;finished={}
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
 assert set(scores['diagnostics'])=={k for k,t in tasks.items() if t['context']=='Q'};unmatched=[]
 for key,d in scores['diagnostics'].items():
  for name,value in d.items():
   if isinstance(value,(int,float)):assert math.isfinite(value),name
  assert 0<=d['alpha']<=1 and d['denominator']>=0 and d['indistinguishable_threshold']>=0 and d['scale_g2']>=0 and d['scale_r2']>=0 and d['scale_rho']>=0
  if d['denominator']<=d['indistinguishable_threshold'] or d['numerator']<=0:assert d['alpha']==0
  else:assert abs(d['alpha']-min(1.,d['numerator']/d['denominator']))<=64*np.finfo(np.float64).eps
  for tag in ['G','null']:
   if d[tag+'_unmatched']:unmatched.append(dict(task=key,control=tag,raw_norm2=d[tag+'_raw_norm2'],norm_error=d[tag+'_norm_error']))
   else:assert d[tag+'_norm_error']<=1e-10
 return c,scores,rows,unmatched

def analyze_effects(scores,rows):
 # Only called after the entire complete, matched numerical screen above.
 from r28_prefix_task_inventory import selected
 wanted={r['id'] for r in rows};gold={};history={}
 for split in ['discovery','selection']:
  with (ROOT/('data/slcd_'+split+'.jsonl')).open(encoding='utf8') as f:
   for line in f:
    identity=selected(line,{'id'})['id']
    if identity in wanted:gold[identity]=selected(line,{'gold'})['gold']
  with (ROOT/('results/r7_'+split+'_development.jsonl')).open(encoding='utf8') as f:
   for line in f:
    identity=selected(line,{'id'})['id']
    if identity in wanted:history[identity]=selected(line,{'baseline_scores','choice_char_lengths','prompt_sha256'})
 methods=list(scores['rows'][0]['candidates'][0]['scores'])+['R6_historical','Fixed_historical'];correct={m:[] for m in methods};predictions=[]
 for rec,row in zip(scores['rows'],rows):
  qid=row['id'];h=history[qid];assert h['choice_char_lengths']==[c['characters'] for c in row['candidates']] and h['prompt_sha256']==hashlib.sha256(row['candidates'][0]['request_pair'][0].encode()).hexdigest();vectors={m:[c['scores'][m] for c in rec['candidates']] for m in methods if m not in ['R6_historical','Fixed_historical']};vectors.update(R6_historical=h['baseline_scores']['M0_R6'],Fixed_historical=h['baseline_scores']['fixed']);p={m:int(np.argmax(v)) for m,v in vectors.items()};y=int(gold[qid]);assert 0<=y<len(row['candidates'])
  for m in methods:correct[m].append(p[m]==y)
  predictions.append(dict(id=qid,gold=y,predictions=p))
 def paired(main,base):
  a,b=correct[main],correct[base];rescue=sum(x and not y for x,y in zip(a,b));damage=sum(y and not x for x,y in zip(a,b));return dict(rescue=rescue,damage=damage,net=rescue-damage)
 retention={};pairs={}
 for base,fixed in [('R6_historical','Fixed_historical'),('R6_matched','Fixed')]:
  unique=[i for i,(a,b) in enumerate(zip(correct[base],correct[fixed])) if a and not b];retention[base]={m:dict(total=len(unique),retained=sum(correct[m][i] for i in unique),ids=[rows[i]['id'] for i in unique]) for m in ['primary','LOO_N1','LOO_N2']};pairs[base]={m:paired(m,base) for m in methods}
 counts={m:sum(v) for m,v in correct.items()};controls=['M3','J2_guidance','G_matched','vocabulary_null','matched_scale'];protected=all(retention[b]['primary']['retained']==retention[b]['primary']['total'] for b in retention);cost=protected and all(pairs[b]['primary']['net']>=0 for b in pairs);strong=protected and all(pairs[b]['primary']['rescue']>=2 and pairs[b]['primary']['damage']==0 for b in pairs) and all(counts['primary']>counts[m] for m in controls) and all(pairs[b][m]['net']>=0 and retention[b][m]['retained']==retention[b][m]['total'] for b in pairs for m in ['LOO_N1','LOO_N2'])
 return dict(n=24,counts=counts,pairs=pairs,retention=retention,strong_screen=strong,cost_review_eligible=cost,controls_ineligible=controls,predictions=predictions,scope='Exploratory reused fixed24 only; matched and historical primitives reported separately, no automatic new GPU')
def main():
 from r28_common import gate
 assert not (ROOT/'results/r28_ANALYSIS.json').exists(),'Official analyzer is single-use'
 spec=gate();c,scores,rows,unmatched=numeric_integrity(spec)
 save('results/r28_LABEL_FREE_INTEGRITY.json',dict(status='passed_complete_label_free_numeric',controls_matched=not unmatched,unmatched=unmatched,protocol_sha256=sha(LOCK),scores_sha256=c['scores_sha256'],audit_sha256=c['audit_sha256'],representative_vectors_sha256=c['representative_vectors_sha256'],native_states_sha256=c['native_states_sha256'],gold_or_argmax_accessed=False))
 if unmatched:save('results/r28_ANALYSIS.json',dict(status='complete_but_unmatched_no_effect_unsealing',unmatched=unmatched,strong_screen=False,cost_review_eligible=False,protocol_sha256=sha(LOCK)));return
 result=analyze_effects(scores,rows);result.update(status='complete_analyzed_once',protocol_sha256=sha(LOCK),scores_sha256=c['scores_sha256']);save('results/r28_ANALYSIS.json',result)
if __name__=='__main__':main()
