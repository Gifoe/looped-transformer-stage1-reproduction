"""Recompute certificates without SciPy/solver status; complete24 only."""
from fractions import Fraction as Q
from r31_oracle_common import ROOT,read,sha,save,gate
from r31_oracle_math import TAU,exact,unpack,NEGATIVE_DUAL_CLIP
from r31_oracle_inputs import load
def certify(row,record):
 p=row['problem'];assert all(record[k]==row[k] for k in ['id','split','gold','background','chains','characters']) and record['saved_late']==row['late'] and record['keys']==p['keys'] and record['zero_keys']==p['zero_keys']
 assert [unpack(v) for v in record['a']]==p['a'] and [[unpack(v) for v in line] for line in record['d']]==p['d'] and [unpack(v) for v in record['B']]==p['B'];y=row['gold'];wrong=[j for j in range(len(p['a'])) if j!=y];assert record['wrong_candidates']==wrong;r=record['proposal'];bits=r['witness'];lower=None;upper=None
 if bits is not None:
  assert len(bits)==len(p['keys']) and all(type(v)==int and v in [0,1] for v in bits)
  # Direct candidate score construction, independently of worker margin helper.
  score=[a+sum((v*x for v,x in zip(d,bits)),Q(0)) for a,d in zip(p['a'],p['d'])];pair=[score[y]-score[j]-p['B'][y]-p['B'][j] for j in wrong];lower=min(pair);assert lower==unpack(r['witness_margin_lower']) and pair==[unpack(v) for v in r['witness_pair_lowers']]
 if r['dual_weights'] is not None:
  w=[unpack(v) for v in r['dual_weights']];assert len(w)==len(wrong) and all(v>=0 for v in w) and sum(w,Q(0))==1;raw=r['raw_dual'];assert len(raw)==len(w) and all(exact(v)>=exact(-NEGATIVE_DUAL_CLIP) for v in raw);clipped=[max(exact(v),Q(0)) for v in raw];total=sum(clipped,Q(0));assert total>0 and w==[v/total for v in clipped]
  constant=sum((v*(p['a'][y]-p['a'][j]+p['B'][y]+p['B'][j]) for v,j in zip(w,wrong)),Q(0));terms=[sum((v*(p['d'][y][k]-p['d'][j][k]) for v,j in zip(w,wrong)),Q(0)) for k in range(len(p['keys']))];upper=constant+sum((max(v,Q(0)) for v in terms),Q(0));assert upper==unpack(r['upper'])
 reachable=lower is not None and lower>TAU;unreachable=upper is not None and upper<=TAU;assert not(reachable and unreachable);classification='certified_reachable' if reachable else 'certified_unreachable_strict_margin' if unreachable else 'undetermined';assert classification==r['classification']
 return classification
def main():
 assert not (ROOT/'results/r31_ORACLE_ANALYSIS.json').exists(),'Only one official certificate analysis'
 gate();sup=read('logs/r31_ORACLE_SUPERVISOR.json');assert sup['status']=='validating' and sup['worker_exit_code']==0 and not sup.get('timed_out',False)
 complete=read('results/r31_ORACLE_COMPLETE.json');raw=read('results/r31_ORACLE_RAW.json');assert complete['status']==raw['status']=='complete' and complete['questions']==24 and complete['fatal'] is None and raw['fatal'] is None and complete['raw_sha256']==sha('results/r31_ORACLE_RAW.json') and complete['protocol_sha256']==raw['protocol_sha256']==sha('configs/r31_SHARED_PREFIX_ORACLE_LOCK.json') and complete['GPU_worker_seconds']==0 and not complete['model_loaded']
 rows,audit=load();assert raw['input_audit']==audit and len(raw['records'])==24;statuses=[certify(row,record) for row,record in zip(rows,raw['records'])];classes=['certified_reachable','certified_unreachable_strict_margin','undetermined'];groups={}
 selectors={'R6_correct':lambda r:r['background']['R6_matched_correct'],'R6_error':lambda r:not r['background']['R6_matched_correct'],'R6_only_gain':lambda r:r['background']['R6_matched_only_gain'],'other_than_R6_only_gain':lambda r:not r['background']['R6_matched_only_gain']}
 for name,test in selectors.items():groups[name]={c:sum(test(row) and s==c for row,s in zip(rows,statuses)) for c in classes}
 errors=[s for row,s in zip(rows,statuses) if not row['background']['R6_matched_correct']];count={c:statuses.count(c) for c in classes};assert sum(count.values())==24
 save('results/r31_ORACLE_ANALYSIS.json',dict(status='complete_certified_pending_clean_supervisor',n=24,counts=count,background_groups=groups,background_groups_overlap=True,rows=[dict(id=r['id'],classification=s,background=r['background']) for r,s in zip(rows,statuses)],R6_error_has_certified_capacity='certified_reachable' in errors,all_R6_errors_certified_unreachable=bool(errors) and all(s=='certified_unreachable_strict_margin' for s in errors),joint_assignment_scope='separate_question_witnesses_compatible' if audit['independent_question_witnesses_jointly_compatible'] else 'undetermined_cross_question_shared_keys',protocol_sha256=sha('configs/r31_SHARED_PREFIX_ORACLE_LOCK.json'),raw_sha256=complete['raw_sha256'],tau='1/100000000',scope='Capacity only, no learned or deployed selector, no accuracy improvement claim. Strict-margin unreachable does not exclude a correct first-index argmax tie. Final validity additionally requires clean supervisor within total180s.',model_loaded=False,GPU_worker_seconds=0))
def final_result():
 s=gate();sup=read('logs/r31_ORACLE_SUPERVISOR.json');assert sup['status']=='complete' and sup['exit_code']==0 and sup['worker_exit_code']==0 and not sup.get('timed_out',False) and not sup['overrun'] and 0<sup['CPU_job_wall_seconds']<=180 and sup['GPU_worker_seconds']==0
 assert sup['analysis_sha256']==sha('results/r31_ORACLE_ANALYSIS.json') and sup['complete_sha256']==sha('results/r31_ORACLE_COMPLETE.json')
 old=s['inherited_GPU_ledgers'];assert len(old)==18 and all(sha(p)==h for p,h in old.items()) and sum(read(p)['used_seconds'] for p in old)==20755.46525352029
 a=read('results/r31_ORACLE_ANALYSIS.json');assert a['status']=='complete_certified_pending_clean_supervisor' and a['raw_sha256']==sha('results/r31_ORACLE_RAW.json')
 return dict(analysis=a,clean_supervisor=sup,validity='complete_clean180s_certificates_joined')
if __name__=='__main__':main()
