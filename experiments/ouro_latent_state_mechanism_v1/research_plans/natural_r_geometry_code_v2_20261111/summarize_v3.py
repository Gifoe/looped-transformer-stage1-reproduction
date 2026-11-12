"""Summary for complete natural R rows; descriptive and full-cohort only."""
import json,hashlib,math,statistics
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr
ROOT=Path('/home/luansiyu/loop-latent');OUT=ROOT/'research_plans/natural_r_geometry_20261111'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def stats(xs):
 v=[float(x) for x in xs if x is not None and math.isfinite(x)];a=np.asarray(v,dtype=np.float64)
 return {'n':len(xs),'finite':len(v),'missing':len(xs)-len(v),'positive':sum(x>0 for x in v),'negative':sum(x<0 for x in v),'zero':sum(x==0 for x in v),'mean':float(a.mean()) if len(v) else None,'quantiles':np.quantile(a,[0,.25,.5,.75,1]).tolist() if len(v) else None}
def corr(x,y):
 pairs=[(float(a),float(b)) for a,b in zip(x,y) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
 if len(pairs)<2:return {'n':len(pairs),'pearson':None,'spearman':None}
 a,b=np.asarray(pairs,dtype=np.float64).T
 if np.ptp(a)==0 or np.ptp(b)==0:return {'n':len(pairs),'pearson':None,'spearman':None}
 return {'n':len(pairs),'pearson':float(pearsonr(a,b).statistic),'spearman':float(spearmanr(a,b).statistic)}
def subgroup(rows):
 out={'n':len(rows),'layers':[]}
 keys=('h1_norm','h2_norm','delta_l2','relative_delta','cosine','norm_ratio','radial_alpha','orthogonal_delta_l2','proxy_fixed_gap_delta','proxy_all_wrong_gap_delta','proxy_common_delta','delta_u_norm','answer_visible_norm','answer_orthogonal_norm','answer_visible_square_fraction')
 for i in range(24):
  ls=[r['layers'][i] for r in rows];v={'layer':i,'stats':{k:stats([x[k] for x in ls]) for k in keys},'corr_fixed':corr([x['proxy_fixed_gap_delta'] for x in ls],[r['fixed_gap'][1]-r['fixed_gap'][0] for r in rows]),'corr_all_wrong':corr([x['proxy_all_wrong_gap_delta'] for x in ls],[r['all_wrong_gap'][1]-r['all_wrong_gap'][0] for r in rows])}
  v['corr_proxy_fixed_with_native_dynamic_margin']=corr([x['proxy_fixed_gap_delta'] for x in ls],[r['dynamic_gap'][1]-r['dynamic_gap'][0] for r in rows]);v['native_dynamic_margin_delta']=stats([r['dynamic_gap'][1]-r['dynamic_gap'][0] for r in rows])
  v['geometry_corr_with_native_fixed']={k:corr([x[k] for x in ls],[r['fixed_gap'][1]-r['fixed_gap'][0] for r in rows]) for k in keys}
  v['geometry_corr_with_native_all_wrong']={k:corr([x[k] for x in ls],[r['all_wrong_gap'][1]-r['all_wrong_gap'][0] for r in rows]) for k in keys}
  v['gold_delta']=stats([x['proxy_delta'][r['gold']] for x,r in zip(ls,rows)]);v['wrong_mean_delta']=stats([statistics.mean(x['proxy_delta'][j] for j in range(len(r['targets'])) if j!=r['gold']) for x,r in zip(ls,rows)]);v['centered_gold_delta']=stats([x['proxy_centered_delta'][r['gold']] for x,r in zip(ls,rows)])
  for k in ('proxy_gap1','proxy_gap2','proxy_advantage','adjacent_gap1_increment','adjacent_gap2_increment','adjacent_advantage_increment'):
   v[k+'_question_equal_weight']=stats([statistics.mean(x[k]) if x[k] is not None else None for x in ls])
  if i==23:
   v['calibration_norm_maxabs']=stats([max(x['calibration']['offline_norm_vs_saved_postnorm_maxabs']) for x in ls]);v['calibration_target_logits_maxabs']=stats([max(abs(y) for z in x['calibration']['offline_proxy_minus_candidate0_target_logits'] for y in z) for x in ls]);v['candidate_request_vs_canonical_target_logits_maxabs']=stats([max(abs(y) for z in x['calibration']['request_minus_candidate0_target_logits'] for y in z) for x in ls])
  out['layers'].append(v)
 return out
def main():
 source=json.loads((OUT/'source.json').read_text());summary={'status':'R_GEOMETRY_DESCRIPTIVE_COMPLETE_PENDING_REVIEW','source_sha256':sha(OUT/'source.json'),'summary_script_sha256':sha(Path(__file__)),'splits':{}}
 for split,n in [('train',1119),('validation',299),('test',1172)]:
  p=OUT/(split+'_rows.jsonl');assert sha(p)==source[split]['rows_sha256'];rows=[json.loads(x) for x in p.open()];assert len(rows)==n and len({r['id'] for r in rows})==n and all(len(r['layers'])==24 for r in rows)
  summary['splits'][split]={'all':subgroup(rows),'non_demo':subgroup([r for r in rows if not r['demo']]),'natural_no_top_tie':subgroup([r for r in rows if r['natural_no_top_tie']]),'groups':{g:subgroup([r for r in rows if r['group']==g]) for g in ('rescue','damage','stable_correct','stable_wrong')},'demo_only':subgroup([r for r in rows if r['demo']]),'non_demo_groups':{g:subgroup([r for r in rows if not r['demo'] and r['group']==g]) for g in ('rescue','damage','stable_correct','stable_wrong')},'non_demo_natural_no_top_tie':subgroup([r for r in rows if not r['demo'] and r['natural_no_top_tie']]),'non_demo_groups_natural_no_top_tie':{g:subgroup([r for r in rows if not r['demo'] and r['natural_no_top_tie'] and r['group']==g]) for g in ('rescue','damage','stable_correct','stable_wrong')}}
 with (OUT/'summary.json').open('x') as f:json.dump(summary,f,sort_keys=True,allow_nan=False);f.write('\n')
 print('SUMMARY_COMPLETE',sha(OUT/'summary.json'))
if __name__=='__main__':main()
