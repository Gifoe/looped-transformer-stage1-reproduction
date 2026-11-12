"""Independent Order003 scoring/factorial/source audit; no executor imports."""
import argparse,json,hashlib,statistics,collections
from fractions import Fraction
from pathlib import Path
from astra_independent_audit import metrics,argmax,group
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--prior',required=True);p.add_argument('--labels',required=True);p.add_argument('--cohort',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.results);prior=Path(a.prior);labels={(r['split'],r['id']):r for r in map(json.loads,Path(a.labels).read_text(encoding='utf-8-sig').splitlines())};cohort=read(a.cohort);expected=set(cohort['ids']);seen=set();fail=[];buckets=collections.defaultdict(list);cases=[];factorials=[];precision=[];headnorm=[]
 required={'q_only','joint_qkv','q_direction_headnorm_matched','q_temperature_headnorm','q_only_norm_only','q_only_unrelated_delta','joint_qkv_norm_only','joint_qkv_unrelated_delta'}
 def check(ok,msg):
  if not ok:fail.append(msg)
 def ties(s):return sum(v==max(s) for v in s)>1
 def verify(r,tag):
  v,s=r['candidate_logits'],r['scores'];check(len(v)==len(s) and max(abs((s[i]-s[0])-(v[i]-v[0])) for i in range(len(v)))<2e-5,tag+' scores');check(argmax(v)==argmax(s) and ties(v)==ties(s),tag+' prediction/ties')
  check(max(abs(v[i]-r['logsumexp_fp32']-s[i]) for i in range(len(v)))<2e-5,tag+' normalization')
 for path in sorted(root.glob('*.results.json')):
  obj=read(path);ident=obj['id'];tag=obj['split']+'/'+ident;check(ident not in seen,tag+' duplicate');seen.add(ident);gold=labels[(obj['split'],ident)]['gold']
  np=path.with_name(path.name.replace('.results.json','.native.json'));nat=read(np);op=prior/np.name;old=read(op);oldr=read(prior/path.name)
  check(obj['native_file_sha256']==sha(np),tag+' native hash');check(nat['prior_native_sha256']==sha(op) and nat['prior_result_sha256']==sha(prior/path.name),tag+' prior hashes');check(nat['input_ids_sha256']==old['input_ids_sha256'],tag+' input identity');check(nat['tensor_sha256']==sha(np.with_suffix('.safetensors')),tag+' tensor hash')
  for k in ['1','2']:check(nat['native'][k]['full_logits_sha256']==old['native'][k]['full_logits_sha256'],tag+' baseline bridge');verify(nat['native'][k],tag+' native'+k)
  rows={r['condition']:r for r in obj['rows']};check(len(rows)==len(obj['rows']),tag+' duplicate condition');check(required-set(obj.get('unavailable',{}))<=set(rows),tag+' condition coverage')
  native=nat['native']['2'];first=nat['native']['1'];fixed=max((i for i in range(len(native['scores'])) if i!=gold),key=lambda i:native['scores'][i]);bm=metrics(native['scores'],gold,fixed);g=group(argmax(first['scores'])==gold,bm['correct']);native_no_tie=not(ties(first['scores'])or ties(native['scores']))
  check(rows['identity']['full_logits_sha256']==native['full_logits_sha256'],tag+' identity');kv=obj['reused_kv_rollback'];oldkv=next(r for r in oldr['rows'] if r['condition']=='target' and r['receiver_loop']==2 and r['donor_loop']==1);check(kv['full_logits_sha256']==oldkv['full_logits_sha256'] and kv['scores']==oldkv['scores'],tag+' KV reuse')
  if obj.get('technical_reused_kv_full_forward_bridge'):check(obj['technical_reused_kv_full_forward_bridge']['full_logits_sha256']==kv['full_logits_sha256'],tag+' KV fullforward bridge')
  for name,r in rows.items():
   verify(r,tag+' '+name);check(r['outside_R_unchanged'],tag+' non-R');check(r['receiver_loop']==2,tag+' receiver')
   if name=='identity':continue
   check((r['donor_id']!=ident) if name.endswith('unrelated_delta') else r['donor_id']==ident,tag+' donor')
   im=metrics(r['scores'],gold,fixed);rec={'id':ident,'split':obj['split'],'condition':name,'group':g,'correct':im['correct'],'rescue':int(not bm['correct'] and im['correct']),'damage':int(bm['correct'] and not im['correct']),'ties':im['ties'],'native_no_tie':native_no_tie}
   for m in ['meanwrong','fixedwrong','dynamicmargin']:rec[m+'_delta']=im[m]-bm[m]
   for strat in ['all',g]:buckets[(obj['split'],name,strat)].append(rec)
   if name in ['q_only','joint_qkv'] and g=='rescue' and not im['correct']:
    nc=rows[name+'_norm_only'];uc=rows[name+'_unrelated_delta'];cases.append({**rec,'control_specific':argmax(nc['scores'])==gold and argmax(uc['scores'])==gold,'no_native_or_target_tie':native_no_tie and not im['ties'],'native_logits':native['candidate_logits'],'target_logits':r['candidate_logits'],'gold':gold})
   if r.get('q_head_receipt'):
    h=r['q_head_receipt'];rn=h['receiver_headnorms'][0];dn=h['donor_headnorms'][0];jn=h['injected_headnorms_after_cast'][0]
    target=rn if name=='q_direction_headnorm_matched' else dn
    headnorm.append({'id':ident,'condition':name,'max_relative_norm_matching_error':max(abs(x-y)/y for x,y in zip(jn,target)) if all(target) else None})
  q=rows['q_only'];joint=rows['joint_qkv'];fr={'id':ident,'split':obj['split'],'group':g}
  cells=[joint,q,kv,native]
  for met in ['meanwrong','fixedwrong','dynamicmargin']:
   vals=[metrics(c['scores'],gold,fixed)[met] for c in cells];exact=[metrics([Fraction(x) for x in c['candidate_logits']],gold,fixed)[met] for c in cells];d=exact[0]-exact[1]-exact[2]+exact[3]
   fr[met]=vals[0]-vals[1]-vals[2]+vals[3];fr[met+'_exact_raw']=float(d);fr[met+'_exact_sign']=(d>0)-(d<0)
  factorials.append(fr)
  if 'q_only_fp32_sdpa_bf16_writeback' in rows:
   fp=rows['q_only_fp32_sdpa_bf16_writeback'];precision.append({'id':ident,'prediction_changed':argmax(fp['scores'])!=argmax(q['scores']),'max_candidate_logit_delta':max(abs(x-y) for x,y in zip(fp['candidate_logits'],q['candidate_logits']))})
 donor_map=read(root/'control_donor_map.json');ids=sorted(seen);check(set(donor_map)==seen,'donor coverage');check(all(donor_map[i]==ids[(n+1)%len(ids)] for n,i in enumerate(ids)),'ID-sorted cyclic donor map')
 check(seen<=expected,'unexpected cohort IDs');summary=[]
 for key,rs in sorted(buckets.items()):
  rec=dict(zip(['split','condition','stratum'],key));rec['n']=len(rs)
  for k in ['correct','rescue','damage','ties']:rec[k]=sum(r[k] for r in rs)
  for met in ['meanwrong','fixedwrong','dynamicmargin']:
   v=[r[met+'_delta'] for r in rs];rec[met+'_delta']={'mean':statistics.mean(v),'median':statistics.median(v),'positive':sum(x>0 for x in v),'negative':sum(x<0 for x in v),'zero':sum(x==0 for x in v)}
  summary.append(rec)
 fs=[]
 for g in ['all','rescue','damage','stable_correct','stable_wrong']:
  rs=[r for r in factorials if g=='all' or r['group']==g]
  if not rs:continue
  f={'group':g,'n':len(rs)}
  for m in ['meanwrong','fixedwrong','dynamicmargin']:f[m]={'mean':statistics.mean(r[m] for r in rs),'exact_positive':sum(r[m+'_exact_sign']>0 for r in rs),'exact_negative':sum(r[m+'_exact_sign']<0 for r in rs),'exact_zero':sum(r[m+'_exact_sign']==0 for r in rs)}
  fs.append(f)
 result={'status':'FAILED' if fail else ('PASS_COMPLETE' if seen==expected else 'PASS_PARTIAL'),'n_expected':len(expected),'n_observed':len(seen),'failures':fail,'summary':summary,'natural_rescue_damage_cases':cases,'factorial_summary':fs,'factorial_rows':factorials,'headnorm_matching':headnorm,'fp32_sdpa_sensitivity':precision}
 Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:result[k] for k in ['status','n_expected','n_observed','failures']}))
 if fail:raise SystemExit(1)
if __name__=='__main__':main()
