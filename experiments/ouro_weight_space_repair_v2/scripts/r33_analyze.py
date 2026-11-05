"""Complete R33 numerical qualification before any label/argmax access."""
import math
import numpy as np
from r33_common import ROOT,LOCK,read,sha,save,gate
from r33_stream import METHODS,STRONG_CONTROLS,native_stream,candidate_readouts
from r30_native_replay import BASE_METHODS,verify_scalar_parity
from r30_analyze import vectors
from r28_analyze import analyze_effects as old_effects
DIAG_KEYS={'mode','n_references','main_norm2','h_exact_zero','c_exact_zero','positive_coordinates','negative_coordinates','scale_g2','scale_r2','scale_rho','scale_unmatched','scale_norm_error','G_unmatched','G_norm_error','G_raw_norm2','G_threshold','null_unmatched','null_norm_error','null_raw_norm2','null_threshold','consensus_unmatched','consensus_norm_error','consensus_raw_norm2','consensus_threshold','LOO_N0_norm2','LOO_N1_norm2','LOO_N2_norm2'}
def scalar_integrity(scores,rows,tasks,uses,vocab_size=49152):
 assert len(scores['rows'])==len(rows)==24;finished={}
 for out,row in zip(scores['rows'],rows):
  assert out['id']==row['id'] and out['split']==row['split'] and len(out['candidates'])==len(row['candidates'])
  for rec,ch in zip(out['candidates'],row['candidates']):
   assert rec['candidate']==ch['candidate'] and rec['characters']==ch['characters']>0 and rec['tokens']==len(ch['Q_target_ids']) and set(rec['pieces'])==set(METHODS)
   for name,piece in rec['pieces'].items():
    size=len(ch['reference_target_ids']['N0']) if name.startswith('N0_') else len(ch['Q_target_ids']);values=np.asarray(piece['target_log_probabilities'],dtype=np.float32);assert values.shape==(size,) and size>0 and np.isfinite(values).all() and np.all(values<=0) and math.isfinite(piece['sum_score']) and float(np.float32(piece['sum_score']))==piece['sum_score']
    exact=math.fsum(float(v) for v in values);scale=math.fsum(abs(float(v)) for v in values);u=np.finfo(np.float32).eps;gamma=(size-1)*u/(1-(size-1)*u);assert abs(piece['sum_score']-exact)<=gamma*scale+np.finfo(np.float64).eps*scale
    finished[(row['id'],ch['candidate'],name)]=piece
 assert candidate_readouts(rows,finished)==scores['rows'];qkeys={k for k,t in tasks.items() if t['context']=='Q'};assert set(scores['diagnostics'])==qkeys==set(uses);unmatched=[]
 for key,d in scores['diagnostics'].items():
  assert set(d)==DIAG_KEYS and d['mode']=='coordinate_actual_common_gain' and type(d['n_references'])==int and d['n_references']==3
  for name,v in d.items():
   if not isinstance(v,str):assert isinstance(v,(int,float,bool)) and math.isfinite(v),name
  assert type(d['h_exact_zero'])==type(d['c_exact_zero'])==bool and d['main_norm2']>=0 and (d['main_norm2']==0)==d['h_exact_zero']
  assert type(d['positive_coordinates'])==type(d['negative_coordinates'])==int and 0<=d['positive_coordinates']+d['negative_coordinates']<=vocab_size and min(d['positive_coordinates'],d['negative_coordinates'])>=0 and d['c_exact_zero']==(d['positive_coordinates']+d['negative_coordinates']==0)
  if d['c_exact_zero']:assert d['h_exact_zero']
  for field in ['scale_g2','scale_r2','scale_rho','scale_norm_error','LOO_N0_norm2','LOO_N1_norm2','LOO_N2_norm2']:assert d[field]>=0
  assert type(d['scale_unmatched'])==bool and d['scale_unmatched']==(not d['h_exact_zero'] and d['scale_g2']==0)
  if d['h_exact_zero']:assert d['scale_rho']==1
  if d['scale_unmatched']:unmatched.append(dict(task=key,control='matched_scale',raw_norm2=d['scale_g2'],norm_error=d['scale_norm_error']))
  else:assert d['scale_norm_error']<=1e-10
  for tag in ['G','null','consensus']:
   assert d[tag+'_raw_norm2']>=0 and d[tag+'_threshold']>=0 and d[tag+'_norm_error']>=0 and type(d[tag+'_unmatched'])==bool and d[tag+'_unmatched']==(not d['h_exact_zero'] and d[tag+'_raw_norm2']<=d[tag+'_threshold'])
   if d[tag+'_unmatched']:unmatched.append(dict(task=key,control=tag,raw_norm2=d[tag+'_raw_norm2'],norm_error=d[tag+'_norm_error']))
   else:assert d[tag+'_norm_error']<=1e-10
  for use in uses[key]:
   position=use['Q_position'];identity=(use['id'],use['candidate']);fixed=finished[identity+('Fixed',)]['target_log_probabilities'][position]
   if d['h_exact_zero']:
    for name in ['primary','G_matched','vocabulary_null','matched_scale','reference_consensus_matched']:assert finished[identity+(name,)]['target_log_probabilities'][position]==fixed
   for j in range(3):
    if d['LOO_N'+str(j)+'_norm2']==0:assert finished[identity+('LOO_N'+str(j),)]['target_log_probabilities'][position]==fixed
 return finished,unmatched

def numeric_integrity(spec):
 c=read('results/r33_COMPLETE.json');sup=read('logs/r33_SUPERVISOR.json');assert sup['status']=='complete' and sup['exit_code']==0 and not sup.get('timed_out',False) and not sup['overrun'] and 0<sup['gpu_worker_seconds']<=200
 old=spec['budget']['inherited_ledger_sha256'];assert len(old)==18
 for p,h in old.items():assert sha(p)==h
 inherited=sum(read(p)['used_seconds'] for p in old);assert inherited==20755.46525352029;book=read('logs/r33_gpu_budget.json');assert book['used_seconds']==sup['gpu_worker_seconds'] and book['inherited_total_seconds']==inherited and book['total_used_seconds']==inherited+book['used_seconds']<=21600
 assert c['status']=='complete' and c['n']==24 and c['body_calls']==1 and c['body_heads']==3 and c['native_head_calls']==7194 and c['total_heads']==7197 and c['groups']==389 and c['fatal'] is None and c['protocol_sha256']==sha(LOCK)
 assert len(c['all_parameter_hashes_before'])==269 and c['all_parameter_hashes_before']==c['all_parameter_hashes_after']==read('results/r28_COMPLETE.json')['all_parameter_hashes_after']
 for field,p in [('scores_sha256','results/r33_SCORES.json'),('audit_sha256','results/r33_EXECUTION_AUDIT.json'),('representative_vectors_sha256','results/r33_REPRESENTATIVE_VECTORS.npz'),('native_parity_sha256','results/r33_NATIVE_PARITY.json')]:assert c[field]==sha(p)
 assert all(math.isfinite(v) and v>=0 for v in c['component_seconds'].values()) and abs(sum(c['component_seconds'].values())-c['measured_components_sum'])<=1e-8 and c['measured_components_sum']<=c['elapsed_seconds']+1e-6<=sup['gpu_worker_seconds']+1e-6
 audit=read('results/r33_EXECUTION_AUDIT.json');cleanup=c['cleanup'];assert cleanup==audit['cleanup'] and cleanup['no_hooks_installed'] and cleanup['all_model_hooks_empty'] and audit['fatal'] is None and audit['native_head_calls']==7194 and audit['body_heads']==3 and audit['total_heads']==7197 and audit['groups']==389
 parity=read('results/r33_NATIVE_PARITY.json');assert parity==audit['native_parity_passes'][0] and len(audit['native_parity_passes'])==2 and parity['native_head_calls']==3597 and parity['historical_saved_fullV_heads_checked']==9
 for p in audit['native_parity_passes']:assert p['status']=='passed_strictzero_native_baseline_scalar_parity' and p['candidates']==96 and not p['gold_read'] and not p['argmax_used']
 graph,tasks,uses,groups,fresh=native_stream();calls=audit['body_calls'];assert len(calls)==1;call=calls[0];task=tasks[fresh];assert call==dict(task=fresh,input_ids_sha256=task['input_ids_sha256'],depth=5,context='Q',length=task['length'],use_cache=False,head_input_shape=[1,1,2048],heads=[1,4,5],no_hooks=True)
 checks=audit['checks'];assert len(checks)==2 and checks[0]['type']=='fresh_Q_exact' and checks[0]['exact'] and checks[0]['fresh']==checks[0]['cached'] and len(checks[0]['fresh']['states'])==5 and set(checks[0]['fresh']['heads'])=={'1','4','5'} and all(r['shape']==[1,1,2048] and r['dtype']=='torch.bfloat16' for r in checks[0]['fresh']['states'])
 rep=checks[1];assert rep['type']=='first_real_common_gain_math' and rep['task'] in tasks and tasks[rep['task']]['context']=='Q' and rep['max_FP32_output_error']<=1e-6 and rep['max_FP64_diag_error']<=1e-10 and rep['q1']['shape']==rep['q4']['shape']==[1,1,49152] and rep['n1']['shape']==rep['n4']['shape']==[3,49152]
 expected_outputs=set(METHODS)-{'native1','native4','native5','adaptive','N0_native1','N0_native4','N0_Fixed'};assert set(rep['outputs'])==expected_outputs and all(v['dtype']=='torch.float32' and v['shape']==[1,1,49152] for v in rep['outputs'].values());vectors('results/r33_REPRESENTATIVE_VECTORS.npz',checks)
 scores=read('results/r33_SCORES.json');rows=read('configs/r28_SANITIZED_REQUESTS.json')['rows'];finished,unmatched=scalar_integrity(scores,rows,tasks,uses);verify_scalar_parity(rows,{k:v for k,v in finished.items() if k[2] in BASE_METHODS},read('results/r28_SCORES.json'));assert len(scores['diagnostics'])==421 and rep['diagnostics']==scores['diagnostics'][rep['task']]
 return c,scores,rows,unmatched

def analyze_effects(scores,rows):
 result=old_effects(scores,rows);pred={r['id']:r for r in result['predictions']}
 for base in result['retention']:
  ids=result['retention'][base]['primary']['ids'];result['retention'][base]['LOO_N0']=dict(total=len(ids),retained=sum(pred[qid]['predictions']['LOO_N0']==pred[qid]['gold'] for qid in ids),ids=ids)
 counts=result['counts'];pairs=result['pairs'];retention=result['retention'];protected=all(retention[b]['primary']['retained']==retention[b]['primary']['total'] for b in retention)
 result['cost_review_eligible']=protected and all(pairs[b]['primary']['net']>=0 for b in pairs);result['strong_screen']=protected and all(pairs[b]['primary']['rescue']>=2 and pairs[b]['primary']['damage']==0 for b in pairs) and all(counts['primary']>counts[m] for m in STRONG_CONTROLS) and all(pairs[b][m]['net']>=0 and retention[b][m]['retained']==retention[b][m]['total'] for b in pairs for m in ['LOO_N0','LOO_N1','LOO_N2']);result['controls_ineligible']=STRONG_CONTROLS;return result
def main():
 assert not (ROOT/'results/r33_ANALYSIS.json').exists(),'Official analyzer single-use'
 spec=gate();c,scores,rows,unmatched=numeric_integrity(spec);save('results/r33_LABEL_FREE_INTEGRITY.json',dict(status='passed_complete_label_free_numeric',controls_matched=not unmatched,unmatched=unmatched,protocol_sha256=sha(LOCK),scores_sha256=c['scores_sha256'],gold_or_argmax_accessed=False))
 if unmatched:save('results/r33_ANALYSIS.json',dict(status='complete_but_unmatched_no_effect_unsealing',unmatched=unmatched,strong_screen=False,cost_review_eligible=False,protocol_sha256=sha(LOCK)));return
 result=analyze_effects(scores,rows);result.update(status='complete_analyzed_once',protocol_sha256=sha(LOCK),scores_sha256=c['scores_sha256']);save('results/r33_ANALYSIS.json',result)
if __name__=='__main__':main()
