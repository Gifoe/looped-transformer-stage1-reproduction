"""Full populations, strict tie accounting and frozen paired comparisons. No fitting."""
import pathlib,json,numpy as np,os,datetime,time
R=pathlib.Path(os.environ.get('CCRU_ROOT','/root/ouro-context-update-sol-20261117'));f=json.loads((R/'FREEZE.json').read_text());selected=json.loads((R/'SELECTED_METHOD.json').read_text())['method'];sign=selected[-1]
rows=[json.loads(l) for l in (R/'ALL_QUESTION_RESULTS.jsonl').read_text().splitlines()] if (R/'ALL_QUESTION_RESULTS.jsonl').exists() else [json.loads(p.read_text()) for p in sorted((R/'results').glob('*.json'))]
assert len(rows)==768 and {q['id'] for q in rows}==set(f['development']+f['confirmation'])
def paired(d,m=4):
 d=np.asarray(d,dtype=float);rng=np.random.default_rng(20261117);v=[]
 for _ in range(20):v.extend(d[rng.integers(0,len(d),(250,len(d)))].mean(1).tolist())
 return {'mean':float(d.mean()),'ci95':np.quantile(v,[.025,.975]).tolist(),'lower_bonferroni4':float(np.quantile(v,.05/m)),'n':len(d)}
def ce(x,g):
 x=np.array(x,dtype=float);a=x.max();return float(a+np.log(np.exp(x-a).sum())-x[g])
summary={};rescues={};prop={};attention={};norms={}
for cohort in ['development','confirmation']:
 qs=[q for q in rows if q['cohort']==cohort];assert len(qs)==len(f[cohort]);methods=list(qs[0]['conditions']);assert all(set(q['conditions'])==set(methods) for q in qs);out={};strict4={};correct={}
 for name in methods:
  out[name]={};norm=[];total=[];matches=[]
  for q in qs:
   r=q['conditions'][name]['intervention'];norm.extend(r['norms']);total.append(r.get('total_displacement_L2',0));z=r.get('dose_matching')
   if z:matches.extend(z['matched'])
  norms[cohort+':'+name]={'questions':len(qs),'token_sites':len(norm),'total_L2_mean':float(np.mean(total)),'per_token_actual_L2_mean':float(np.mean([z[3] for z in norm])) if norm else 0,'max_actual_relative':max((z[3]/z[4] for z in norm),default=0),'clipped_tokens':sum(z[2]<1 for z in norm),'random_matched_sites':sum(matches),'random_sites':len(matches),'sum_capture_seconds':sum(q['conditions'][name]['seconds'] for q in qs),'peak_allocated_bytes':max(q['conditions'][name]['peak_allocated_bytes'] for q in qs)}
  for depth in ['1','2','3','4']:
   a=[q['conditions'][name]['scores'][depth] for q in qs];b=[q['conditions']['N']['scores'][depth] for q in qs];ac=np.array([x['prediction']==q['gold'] for x,q in zip(a,qs)],dtype=int);bc=np.array([x['prediction']==q['gold'] for x,q in zip(b,qs)],dtype=int);unique=np.array([len(x['ties'])==len(y['ties'])==1 for x,y in zip(a,b)]);delta=ac-bc;strict=delta*unique
   if depth=='4':strict4[name]=strict;correct[name]=ac
   out[name][depth]={'n':len(qs),'correct':int(ac.sum()),'accuracy':float(ac.mean()),'rescue':int((delta>0).sum()),'damage':int((delta<0).sum()),'strict_rescue':int((strict>0).sum()),'strict_damage':int((strict<0).sum()),'strict_net':int(strict.sum()),'ties':sum(len(z['ties'])>1 for z in a),'paired_ordinary':paired(delta),'paired_strict':paired(strict),'mean_candidate_CE':float(np.mean([ce(z['candidate_logits'],q['gold']) for z,q in zip(a,qs)])),'mean_gold_margin':float(np.mean([z['candidate_logits'][q['gold']]-max(x for i,x in enumerate(z['candidate_logits']) if i!=q['gold']) for z,q in zip(a,qs)]))}
   rescues[cohort+':'+name+':K'+depth]=[{'id':q['id'],'gold':q['gold'],'native':y,'modified':x,'effect':'rescue' if v>0 else 'damage','all_native_loops_wrong':all(q['conditions']['N']['scores'][str(d)]['prediction']!=q['gold'] for d in range(1,5))} for q,x,y,v in zip(qs,a,b,strict) if v]
 summary[cohort]={'methods':out}
 if cohort=='confirmation':
  contrasts={'N':paired(strict4[selected])};contrasts.update({n:paired(strict4[selected]-strict4[n]) for n in ['R'+sign,'P'+sign,'Random'+sign]});summary[cohort]['primary_contrasts']=contrasts;summary[cohort]['gate_pass']=all(v['lower_bonferroni4']>0 for v in contrasts.values()) and out[selected]['4']['correct']>out['N']['4']['correct']
 trajectory=[]
 for q in qs:
  a=q['conditions'][selected]['scores'];b=q['conditions']['N']['scores'];g=q['gold'];k2=a['2']['prediction']==g and b['2']['prediction']!=g;k4=a['4']['prediction']==g and b['4']['prediction']!=g
  probs={}
  for dep in ['2','3','4']:
   x=np.array(a[dep]['candidate_logits'],dtype=float);y=np.array(b[dep]['candidate_logits'],dtype=float);p=np.exp(x-x.max());p/=p.sum();p0=np.exp(y-y.max());p0/=p0.sum();probs[dep]={'native':p0.tolist(),'modified':p.tolist(),'KL_modified_vs_native':float(np.sum(p*np.log(np.maximum(p,1e-300)/np.maximum(p0,1e-300))))}
  trajectory.append({'id':q['id'],'K2_rescue':k2,'selected_K4_correct':a['4']['prediction']==g,'K4_rescue':k4,'strict_K4_rescue':k4 and len(a['4']['ties'])==len(b['4']['ties'])==1,'new_correct_all_native_loops_wrong':a['4']['prediction']==g and all(b[str(d)]['prediction']!=g for d in range(1,5)),'candidate_distributions':probs})
 prop[cohort]={'rows':trajectory,'K2_rescues':sum(z['K2_rescue'] for z in trajectory),'K2_rescues_correct_at_K4':sum(z['K2_rescue'] and z['selected_K4_correct'] for z in trajectory),'new_correct_all_native_loops_wrong':sum(z['new_correct_all_native_loops_wrong'] for z in trajectory)}
 for group in ['all','strict_K4_rescue','damage','other']:
  pairs=[]
  for q,z in zip(qs,trajectory):
   a=q['conditions'][selected]['scores']['4'];b=q['conditions']['N']['scores']['4'];damage=a['prediction']!=q['gold'] and b['prediction']==q['gold'] and len(a['ties'])==len(b['ties'])==1
   if group=='all' or (group=='strict_K4_rescue' and z['strict_K4_rescue']) or (group=='damage' and damage) or (group=='other' and not z['strict_K4_rescue'] and not damage):pairs.append(q)
  attention[cohort+':'+group]={'n':len(pairs),'layers':{key:{'selected_mean_delta_L2':float(np.mean([q['conditions'][selected]['attention_change_vs_native'][key]['delta_L2'] for q in pairs])),'random_mean_delta_L2':float(np.mean([q['conditions']['Random'+sign]['attention_change_vs_native'][key]['delta_L2'] for q in pairs]))} for key in [f'{d}:{l}' for d in [2,3,4] for l in range(10,17)]} if pairs else {}}
for cohort in ['development','confirmation']:
 qs=[q for q in rows if q['cohort']==cohort]
 for key in [f'{d}:{l}' for d in [2,3,4] for l in range(10,17)]:
  values=np.array([q['conditions'][selected]['attention_change_vs_native'][key]['delta_L2']-q['conditions']['Random'+sign]['attention_change_vs_native'][key]['delta_L2'] for q in qs]);rng=np.random.default_rng(20261117);samples=[]
  for _ in range(20):samples.extend(values[rng.integers(0,len(qs),(250,len(qs)))].mean(1).tolist())
  attention[cohort+':all']['layers'][key]['paired_selected_minus_random_delta_norm']={'mean':float(values.mean()),'descriptive_ci95':np.quantile(samples,[.025,.975]).tolist(),'bonferroni21_two_sided_ci':np.quantile(samples,[.025/21,1-.025/21]).tolist(),'scope':'21 observational layer/depth contrasts per cohort; not causal or an accuracy gate'}
for n,v in [('SUMMARY.json',summary),('STRICT_NON_TIE_RESCUES.json',rescues),('PROPAGATION.json',prop),('ATTENTION_ASSOCIATIONS.json',attention),('NORMS_AND_COSTS.json',norms)]: (R/n).write_text(json.dumps(v,indent=2))
if not (R/'ALL_QUESTION_RESULTS.jsonl').exists():(R/'ALL_QUESTION_RESULTS.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows))
print('ANALYSIS_COMPLETE',selected,'CONFIRMATION_PASS',summary['confirmation']['gate_pass'])
