"""Question-level historical cross-tabs, figures and final resource join; no model."""
from pathlib import Path
import csv,json,collections,itertools,hashlib
import numpy as np
O=Path(__file__).resolve().parent
read=lambda p:json.loads((O/p).read_text(encoding='utf8'))
def rows(p):return list(csv.DictReader((O/p).open(encoding='utf8',newline='')))
def b(x):return x=='True'
def save(p,x):(O/p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def main():
 s=rows('permutation_stability.csv');by={}
 for r in s:by.setdefault(r['id'],{})[int(r['K'])]=r
 tr={r['id']:r for r in rows('k1_k4_paired_transitions.csv') if r['family']=='cyclic_mean'};tab={k:[] for k in itertools.product([False,True],repeat=4)};sub=[];changes={'rescue':collections.Counter(),'damage':collections.Counter()}
 for q,ks in by.items():
  a,c=ks[1],ks[4];key=(b(a['unanimous']),b(c['unanimous']),b(tr[q]['K1_correct']),b(tr[q]['K4_correct']));tab[key].append(q);delta=float(c['pairwise_content_agreement'])-float(a['pairwise_content_agreement']);kind='increase' if delta>0 else 'decrease' if delta<0 else 'same'
  if tr[q]['transition'] in changes:changes[tr[q]['transition']][kind]+=1
  sub.append(dict(id=q,K1_unanimous=key[0],K4_unanimous=key[1],K1_cyclic_correct=key[2],K4_cyclic_correct=key[3],agreement_change=kind,transition=tr[q]['transition']))
 stable_rescue=tab[(True,True,False,True)];stable_damage=tab[(True,True,True,False)];assert len(stable_rescue)==47 and len(stable_damage)==1
 save('stability_correctness_cross_tab.json',dict(groups=[dict(K1_unanimous=k[0],K4_unanimous=k[1],K1_cyclic_correct=k[2],K4_cyclic_correct=k[3],n=len(v),ids=v) for k,v in tab.items()],stable_wrong_to_stable_correct=stable_rescue,stable_correct_to_stable_wrong=stable_damage,cyclic_rescue_damage_agreement_change={k:dict(v) for k,v in changes.items()},scope='Question-level descriptive cross-tab. Same stability state does not imply same answer, and does not prove content-comparison circuitry.'))
 with (O/'stability_correctness_question_table.csv').open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(sub[0]));w.writeheader();w.writerows(sub)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
 summary=read('stability_summary.json');paired=read('paired_statistics.json');fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained');fig.suptitle('n=1172; K2/K3 unobserved; lines connect observed endpoints only',fontsize=11);x=np.arange(2);width=.32
 ax=axs[0,0]
 for i,fam in enumerate(['joint','cyclic_mean']):
  r=paired[i];ax.plot([1,4],np.array([r['K1_correct'],r['K4_correct']])/1172*100,'o-',label=fam)
 ax.set(title='A  Historical accuracy',xlabel='Observed native K',ylabel='Correct (%)',xticks=[1,4]);ax.legend(frameon=False)
 ax=axs[0,1];ax.bar(x-width/2,[r['rescue'] for r in paired],width,label='Rescue');ax.bar(x+width/2,[r['damage'] for r in paired],width,label='Damage');ax.set(title='B  Paired K1 to K4 transitions',ylabel='Questions',xticks=x,xticklabels=['Joint','Cyclic mean']);ax.set_ylim(0,535);ax.legend(frameon=False,loc='upper right',ncol=2)
 for container in ax.containers:ax.bar_label(container,padding=3)
 ax=axs[1,0];ax.plot([1,4],[r['unanimous']/1172*100 for r in summary['per_depth']],'o-',label='Unanimous questions');ax.plot([1,4],[r['mean_pairwise_agreement']*100 for r in summary['per_depth']],'o-',label='Mean pairwise agreement');ax.set(title='C  Cyclic-rotation content consistency',xlabel='Observed native K',ylabel='Agreement (%)',xticks=[1,4]);ax.legend(frameon=False)
 ax=axs[1,1]
 for name,label in [('mean_mapped_score_variance','Raw scores'),('mean_centered_score_variance','Centered scores')]:
  ax.plot([1,4],[np.mean([float(r[name]) for r in s if int(r['K'])==k]) for k in [1,4]],'o-',label=label)
 ax.set(title='D  Score variance does not decrease',xlabel='Observed native K',ylabel='Across-rotation variance (log-prob sum)²',xticks=[1,4]);ax.legend(frameon=False)
 fig.savefig(O/'historical_decision_audit.png',dpi=180);fig.savefig(O/'historical_decision_audit.svg');plt.close(fig)
 # Public receipts omit private local transport-coordinate request paths.
 for original,public in [('official_server_cpu_replay.json','official_environment_exact_replay.json'),('runtime_metadata_raw.json','runtime_metadata.json'),('a4_SERVER_PREFLIGHT.json','a4_SERVER_PREFLIGHT_PUBLIC.json'),('a4_START_RECEIPT_RAW.json','a4_START_RECEIPT.json'),('a4_OFFICIAL_ANALYZER_RECEIPT_RAW.json','a4_OFFICIAL_ANALYZER_RECEIPT.json')]:
  if not (O/original).exists():
   assert (O/public).exists() and read(public).get('exit_code')==0;continue
  d=read(original);out={k:v for k,v in d.items() if k!='request'}
  if original in ['official_server_cpu_replay.json','runtime_metadata_raw.json']:out['result']=json.loads(out.pop('stdout'))
  save(public,out)
 if (O/'live_resource_cache_readonly.json').exists():
  d=read('live_resource_cache_readonly.json');v=json.loads(d['stdout']);save('live_resource_cache_preflight.json',dict(observed_unix=v['observed_unix'],GPU_query=v['GPU_query'],our_worker_count=0,cache_namespace_search=v['cache_namespace_search'],old_ledger_total=v['total'],model_loaded=False,forward_or_head=False,scope='Metadata/name inventory, not tensor payload audit. Prior R33 structural cache inventory separately bound. No joint state cache identified.'))
 else:assert (O/'live_resource_cache_preflight.json').exists() and not read('live_resource_cache_preflight.json')['model_loaded']
 p=read('existing_score_parity.json');p['original_environment_official_replay']='official_environment_exact_replay.json';p['Windows_projection_replay']='Failed strict old R6 reconstruction at FP64 maxdiff1.1102230246251565e-16 on one inspected row; original server complete exact all14 passed; attempted logs retained. No tolerance substituted for original gate.';save('existing_score_parity.json',p)
 ledger=read('compute_ledger.json');ledger.pop('model_loaded',None);ledger.update(CPU_audit_model_loaded=False,model_loaded_in_new_attempt=True);actual=read('a4_gpu_budget.json');ledger.update(CPU_audit_new_GPU_worker_seconds=0,new_model_attempt_ledger='a4_gpu_budget.json',new_model_attempt_ledger_sha256=hashlib.sha256((O/'a4_gpu_budget.json').read_bytes()).hexdigest(),new_GPU_worker_seconds=actual['used_seconds'],total_GPU_worker_seconds=actual['total_used_seconds'],remaining_seconds=actual['remaining_seconds'],A4_state='parity_only_runtime_not_admitted_64_no_forward');save('compute_ledger.json',ledger)

 coverage=read('depth_readout_coverage.json');coverage['new_fresh_parity_only']=dict(questions=2,rotations=8,native_K=[1,2,3,4],independent_depth2_3_checks=16,state_head_target_LP_exact=True,model_calls=24,native_head_calls=72,cohort64_model_calls=0,full576_model_calls=0,interpretation='Technical validation only; no64 or full-test intermediate behavior inferred.')
 coverage['historical_development_auxiliary']=dict(questions=576,available_native_K=[1,4,5],executed_depth=5,source='Historical R10 depth5 records, not merged into R11 depth4 trajectories; no newK5/K6.')
 coverage['cache_inventory']=dict(current_metadata='live_resource_cache_preflight.json',prior_structural_inventory='../results/r33_RESOURCE_CACHE_PREFLIGHT_PUBLIC.json',prior_sha256=hashlib.sha256((O.parent/'results/r33_RESOURCE_CACHE_PREFLIGHT_PUBLIC.json').read_bytes()).hexdigest(),scope='Old answer-body native1..6 caches are not Joint label-prompt states; no compatible Joint cache identified. No intermediate imputation.')
 save('depth_readout_coverage.json',coverage)
 predictions=rows('loopwise_predictions.csv');margin_summary={}
 for family in ['joint','cyclic_mean']:
  a=[r for r in predictions if r['family']==family and r['K']=='1'];b4=[r for r in predictions if r['family']==family and r['K']=='4'];delta=[float(y['gold_margin'])-float(x['gold_margin']) for x,y in zip(a,b4)]
  margin_summary[family]=dict(mean_gold_margin_K1=float(np.mean([float(r['gold_margin']) for r in a])),mean_gold_margin_K4=float(np.mean([float(r['gold_margin']) for r in b4])),increased=sum(x>0 for x in delta),unchanged=sum(x==0 for x in delta),decreased=sum(x<0 for x in delta))
 save('gold_margin_descriptive_summary.json',margin_summary)
 print(json.dumps(dict(stable_rescue=len(stable_rescue),stable_damage=len(stable_damage),agreement_changes={k:dict(v) for k,v in changes.items()},totalGPU=actual['total_used_seconds'])))
if __name__=='__main__':main()
