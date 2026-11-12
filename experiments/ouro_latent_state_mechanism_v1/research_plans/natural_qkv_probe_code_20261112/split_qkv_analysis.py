"""Descriptive train-only QKV analysis of previously CPU-audited artifacts."""
import json,gzip,hashlib,math,os,sys
from fractions import Fraction
from pathlib import Path
import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file
from source_input import ROOT,sha
SPLIT=sys.argv[1];assert SPLIT in ('validation','test');NQ=299 if SPLIT=='validation' else 1172;BASE=ROOT/('research_plans/natural_qkv_'+SPLIT+'_preliminary_20261112');RUN=ROOT/'research_plans/natural_qkv_full_20261112';SPEC=ROOT/'research_plans/NATURAL_QKV_TRAIN_ANALYSIS_SPEC_20261112.md'
N=('F11','F21','F12','F22')
def require(x,s):
 if not x:raise ValueError(s)
def metrics(a):
 return {'q_kv1':a['F21']-a['F11'],'q_kv2':a['F22']-a['F12'],'kv_q1':a['F12']-a['F11'],'kv_q2':a['F22']-a['F21'],'interaction':a['F22']-a['F21']-a['F12']+a['F11']}
def dstat(x):
 a=np.asarray(x,dtype=np.float64);return {'N':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'q05':float(np.quantile(a,.05)),'q25':float(np.quantile(a,.25)),'q75':float(np.quantile(a,.75)),'q95':float(np.quantile(a,.95)),'positive':int((a>0).sum()),'negative':int((a<0).sum()),'zero':int((a==0).sum())}
def vec(x):
 norms={n:float(np.linalg.norm(v)) for n,v in x.items()};diff=metrics(x);dn={k:float(np.linalg.norm(v)) for k,v in diff.items()};co={}
 for label,a,b in [('q','q_kv1','q_kv2'),('kv','kv_q1','kv_q2')]:
  dot=float(np.dot(diff[a],diff[b]));co[label]={'dot':dot,'cosine':dot/(dn[a]*dn[b]) if dn[a] and dn[b] else None}
 return {'corner_norms':norms,'difference_norms':dn,'pairwise':co}
def main():
 require(sha(ROOT/'research_plans/QKV_TRAIN_INFORMED_PROSPECTIVE_CHECKS_20261112.md')=='a5d1fe62e91af1b6759a54edd55105996967009d1d5e39869fb38bdb24423376','prospective freeze');require(sha(SPEC)=='08d90dc21bf2aab65e65e9a775695762398a45c711631bda02f60e2533ddc2d2','spec');cov=json.loads((BASE/'coverage.json').read_text());require(cov['status']=='CPU_SOURCE_ENDPOINT_PASS_INDEPENDENT_CUDA_PENDING' and cov['count']==NQ and sha(BASE/'rows.jsonl.gz')==cov['rows_sha256'],'coverage')
 with gzip.open(BASE/'rows.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
 ledger={r['id']:r for r in cov['questions']};require(len(rows)==NQ and len(ledger)==NQ and {r['id'] for r in rows}==set(ledger) and all(r['metadata_sha256']==ledger[r['id']]['metadata_sha256'] and r['tensor_sha256']==ledger[r['id']]['tensor_sha256'] for r in rows),'row IDs/hashes');weights=ROOT/'research_plans/natural_r_geometry_20261111/target_weights.safetensors';require(sha(weights)=='472f6ab25164a1f2b39fe92432120d6b4176125c086d67bd6174aa95fe8fd647','frozen weights');frozen=load_file(str(weights));index={int(v):i for i,v in enumerate(frozen['target_ids'].tolist())};require(len(index)==5 and tuple(frozen['weight_rows'].shape)==(5,2048) and tuple(frozen['norm_weight'].shape)==(2048,),'frozen weights shape');out=BASE/'analysis_rows.jsonl'
 with out.open('x') as f:
  gamma=frozen['norm_weight'].float().numpy().astype(np.float64)
  for row in rows:
   key=SPLIT+'_'+hashlib.sha256(row['id'].encode()).hexdigest();tp=RUN/(key+'.safetensors');require(sha(tp)==row['tensor_sha256'],'tensor drift');t=load_file(str(tp));gold=row['gold'];wrong=[j for j in range(len(row['targets'])) if j!=gold];w=np.stack([frozen['weight_rows'][index[i]].float().numpy().astype(np.float64) for i in row['targets']]);direction=(w[gold]-w[wrong].mean(axis=0))*gamma;fixed=(w[gold]-w[row['fixed_wrong']])*gamma
   gaps={n:sum(row['scores'][n][gold]-row['scores'][n][j] for j in wrong)/len(wrong) for n in N};fx={n:row['scores'][n][gold]-row['scores'][n][row['fixed_wrong']] for n in N};exact={n:sum((Fraction.from_float(row['target_logits'][n][gold])-Fraction.from_float(row['target_logits'][n][j]) for j in wrong),Fraction())/len(wrong) for n in N};fixedex={n:Fraction.from_float(row['target_logits'][n][gold])-Fraction.from_float(row['target_logits'][n][row['fixed_wrong']]) for n in N};sm={n:t[n+'_sdpa_R'].float().numpy().astype(np.float64).ravel() for n in N};am={n:t[n+'_a_R'].float().numpy().astype(np.float64).ravel() for n in N};proj={n:float(np.dot(direction,am[n])) for n in N};projf={n:float(np.dot(fixed,am[n])) for n in N};fm=metrics(gaps);fem=metrics(exact);ffm=metrics(fixedex);phi_q=(fm['q_kv1']+fm['q_kv2'])/2;phi_k=(fm['kv_q1']+fm['kv_q2'])/2;require(abs(phi_q+phi_k-(gaps['F22']-gaps['F11']))<1e-10,'accounting')
   record={'id':row['id'],'metadata_sha256':row['metadata_sha256'],'tensor_sha256':row['tensor_sha256'],'demo':row['demo'],'group':row['group'],'natural_no_top_tie':row['natural_no_top_tie'],'scores':row['scores'],'natural_scores':row['natural_scores'],'gold':gold,'targets':row['targets'],'fixed_wrong':row['fixed_wrong'],'predictions':row['predictions'],'top_ties':row['top_ties'],'D':gaps['F12']-gaps['F21'],'F':gaps,'F_fixed':fx,'F_difference':fm,'F_fixed_difference':metrics(fx),'dynamic_gap':{n:row['scores'][n][gold]-max(row['scores'][n][j] for j in wrong) for n in N},'F_exact_differences':{k:str(v) for k,v in fem.items()},'F_exact_signs':{k:(v>0)-(v<0) for k,v in fem.items()},'F_fixed_exact_signs':{k:(v>0)-(v<0) for k,v in ffm.items()},'phiQ':phi_q,'phiKV':phi_k,'sdpa':vec(sm),'a':vec(am),'a_projection':proj,'a_projection_difference':metrics(proj),'a_fixed_projection_difference':metrics(projf)};f.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
 grouped={}
 with out.open() as f:data=[json.loads(line) for line in f]
 groups={'all':data,'non_demo':[r for r in data if not r['demo']],'demo':[r for r in data if r['demo']],'natural_no_top_tie':[r for r in data if r['natural_no_top_tie']],'non_demo_natural_no_top_tie':[r for r in data if not r['demo'] and r['natural_no_top_tie']]}
 for g in ('rescue','damage','stable_correct','stable_wrong'):groups[g]=[r for r in data if r['group']==g];groups['non_demo_'+g]=[r for r in data if r['group']==g and not r['demo']];groups['no_tie_'+g]=[r for r in data if r['group']==g and r['natural_no_top_tie']];groups['non_demo_no_tie_'+g]=[r for r in data if r['group']==g and not r['demo'] and r['natural_no_top_tie']]
 def edges(rs):
  result={}
  for label,left,right in [('Q_KV1','F11','F21'),('Q_KV2','F12','F22'),('KV_Q1','F11','F12'),('KV_Q2','F21','F22'),('endpoint','F11','F22')]:
   v={'unchanged':0,'changed':0,'correct_to_wrong':0,'wrong_to_correct':0,'wrong_to_other_wrong':0,'baseline_tie':0,'treatment_tie':0}
   for r in rs:
    a,b=r['predictions'][left],r['predictions'][right];g=r['gold'];v['unchanged' if a==b else 'changed']+=1;v['correct_to_wrong']+=int(a==g and b!=g);v['wrong_to_correct']+=int(a!=g and b==g);v['wrong_to_other_wrong']+=int(a!=g and b!=g and a!=b);v['baseline_tie']+=int(r['top_ties'][left]);v['treatment_tie']+=int(r['top_ties'][right])
   v['net_recovery']=v['wrong_to_correct']-v['correct_to_wrong'];result[label]=v
  return result
 for name,rs in groups.items():
  if not rs:
   grouped[name]={'N':0};continue
  grouped[name]={'N':len(rs),'F_difference':{k:dstat([r['F_difference'][k] for r in rs]) for k in metrics({n:0. for n in N})},'a_projection_difference':{k:dstat([r['a_projection_difference'][k] for r in rs]) for k in metrics({n:0. for n in N})},'sdpa_difference_norms':{k:dstat([r['sdpa']['difference_norms'][k] for r in rs]) for k in metrics({n:0. for n in N})},'a_difference_norms':{k:dstat([r['a']['difference_norms'][k] for r in rs]) for k in metrics({n:0. for n in N})}}
  grouped[name]['D']=dstat([r['D'] for r in rs]);grouped[name]['correct_counts']={n:sum(r['predictions'][n]==r['gold'] for r in rs) for n in N};grouped[name]['edges']=edges(rs)
  grouped[name]['fixed_difference']={k:dstat([r['F_fixed_difference'][k] for r in rs]) for k in metrics({n:0. for n in N})}
  grouped[name]['fraction_signs']={k:{'positive':sum(r['F_exact_signs'][k]>0 for r in rs),'negative':sum(r['F_exact_signs'][k]<0 for r in rs),'zero':sum(r['F_exact_signs'][k]==0 for r in rs),'production_sign_disagreements':sum(((r['F_difference'][k]>0)-(r['F_difference'][k]<0))!=r['F_exact_signs'][k] for r in rs)} for k in metrics({n:0. for n in N})}
  for space in ('sdpa','a'):
   grouped[name][space+'_corner_norms']={k:dstat([r[space]['corner_norms'][k] for r in rs]) for k in N}
   grouped[name][space+'_cosines']={k:({'defined':len(v),'undefined':len(rs)-len(v),'distribution':dstat(v) if v else None}) for k in ('q','kv') for v in [[r[space]['pairwise'][k]['cosine'] for r in rs if r[space]['pairwise'][k]['cosine'] is not None]]}
  z={}
  for r in rs:
   key='s%s_a%s_F%s'%(int(r['sdpa']['difference_norms']['interaction']==0),int(r['a']['difference_norms']['interaction']==0),int(r['F_exact_signs']['interaction']==0));z[key]=z.get(key,0)+1
  grouped[name]['interaction_exact_zero_joint']=z
 require(sha(weights)=='472f6ab25164a1f2b39fe92432120d6b4176125c086d67bd6174aa95fe8fd647' and all(sha(RUN/(SPLIT+'_'+hashlib.sha256(r['id'].encode()).hexdigest()+'.json'))==r['metadata_sha256'] and sha(RUN/(SPLIT+'_'+hashlib.sha256(r['id'].encode()).hexdigest()+'.safetensors'))==r['tensor_sha256'] for r in rows) and sha(BASE/'rows.jsonl.gz')==cov['rows_sha256'] and sha(SPEC)=='08d90dc21bf2aab65e65e9a775695762398a45c711631bda02f60e2533ddc2d2','final source drift')
 result={'prospective_sha256':sha(ROOT/'research_plans/QKV_TRAIN_INFORMED_PROSPECTIVE_CHECKS_20261112.md'),'status':'SPLIT_EXPLORATORY_CPU_PENDING_INDEPENDENT_CUDA','groups':grouped,'rows_sha256':sha(out),'coverage_sha256':sha(BASE/'coverage.json'),'spec_sha256':sha(SPEC),'script_sha256':sha(Path(__file__)),'model_sha256':'58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af','model_hash_scope':'frozen target W rows and gamma, no whole-model rehash','target_weights_sha256':sha(weights),'caveats':'Internal projected states are descriptive and are not final-logit causal attributions. Earlier natural and intervention results were exposed.'}
 with (BASE/'summary.json').open('x') as f:json.dump(result,f,sort_keys=True,allow_nan=False);f.write('\n')
 print('TRAIN_QKV_SUMMARY_COMPLETE',sha(BASE/'summary.json'),flush=True)
if __name__=='__main__':main()
