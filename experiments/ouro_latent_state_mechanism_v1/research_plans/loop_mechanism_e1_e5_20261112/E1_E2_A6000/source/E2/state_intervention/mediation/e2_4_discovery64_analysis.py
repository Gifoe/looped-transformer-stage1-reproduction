"""Analyze pre-audited E2.4 64-case four-corner outcomes; no model calls."""
import json,gzip,pathlib,hashlib,statistics,collections,datetime
from fractions import Fraction
from safetensors.torch import load_file
R=pathlib.Path('/home/luansiyu/loop-latent');B=R/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E2';P=E/'state_intervention/mediation/discovery64';OUT=E/'state_intervention/mediation/discovery64_analysis';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((P/'CONTROL_MATH_AUDIT.json').read_text());b=json.loads((P/'CUDA_REPLAY.json').read_text());assert a['status']=='E2_4_DISCOVERY64_INDEPENDENT_CONTROL_MATH_PASSED' and b['status']=='E2_4_DISCOVERY64_INDEPENDENT_DOUBLEHOOK_CUDA_REPLAY_PASSED';pending=json.loads((P/'PENDING.json').read_text());assert len(pending['cases'])==64 and a['pending_sha256']==sha(P/'PENDING.json') and b['source_pending_sha256']==sha(P/'PENDING.json')
with gzip.open(E/'E2_latent_transition_rows.jsonl.gz','rt') as f:source={(r['split'],r['id']):r for r in map(json.loads,f)}
rows=[];conditions=('C00','C11','C10','C01','full_norm','full_unrelated');contrasts={'attention_at_x2':{'C00':1,'C01':-1},'attention_at_x1':{'C10':1,'C11':-1},'residual_at_a2':{'C00':1,'C10':-1},'residual_at_aprime':{'C01':1,'C11':-1},'interaction':{'C00':1,'C10':-1,'C01':-1,'C11':1},'fullnorm_vs_C00':{'full_norm':1,'C00':-1},'unrelated_vs_C00':{'full_unrelated':1,'C00':-1},'fullnorm_vs_C11':{'full_norm':1,'C11':-1},'unrelated_vs_C11':{'full_unrelated':1,'C11':-1}}
def metric(vals,g,fixed):
 wrong=[i for i in range(len(vals)) if i!=g];best=max(vals);ties=[i for i,x in enumerate(vals) if x==best];return {'scores':vals,'pred':ties[0],'ties':ties,'correct':ties[0]==g,'allwrong':vals[g]-statistics.mean(vals[j] for j in wrong),'fixedwrong':vals[g]-vals[fixed],'dynamic':vals[g]-max(vals[j] for j in wrong)}
def exact(vals,g,wrong,fixed,field):
 z=[Fraction.from_float(x) for x in vals];return z[g]-(sum((z[i] for i in wrong),Fraction())/len(wrong) if field=='allwrong' else z[fixed] if field=='fixedwrong' else max(z[i] for i in wrong))
for case in pending['cases']:
 key=hashlib.sha256((case['split']+'\0'+case['id']).encode()).hexdigest();mp=P/(key+'.json');tp=P/(key+'.safetensors');assert sha(mp)==case['metadata_sha256'] and sha(tp)==case['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));r=source[(case['split'],case['id'])];assert m['targets']==r['targets'];g=r['gold'];wrong=[i for i in range(len(r['targets'])) if i!=g];fixed=max(wrong,key=lambda j:r['native_scores'][0][j]);native=metric(r['native_scores'][1],g,fixed);cs={}
 for name in conditions:
  z=m['conditions'][name]
  if z['status']!='OK':cs[name]={'status':'UNDEFINED'};continue
  scores=z['candidate_full_vocab_logprob'];z2=metric(scores,g,fixed);z2['status']='OK';z2['raw_target_bf16_logits']=[float(t[name+'_logits'][0,i]) for i in r['targets']];z2['became_wrong']=native['correct'] and not z2['correct'];z2['became_correct']=not native['correct'] and z2['correct'];cs[name]=z2
 assert cs['C00']['scores']==r['native_scores'][1];effect={}
 for name,terms in contrasts.items():
  if any(cs[k]['status']!='OK' for k in terms):effect[name]={'status':'UNDEFINED'};continue
  v={f:sum(coef*cs[k][f] for k,coef in terms.items()) for f in ('allwrong','fixedwrong','dynamic')};ex={f:sum((coef*exact(cs[k]['raw_target_bf16_logits'],g,wrong,fixed,f) for k,coef in terms.items()),Fraction()) for f in ('allwrong','fixedwrong','dynamic')};v.update({'status':'OK','exact_bf16':{f:str(val) for f,val in ex.items()},'exact_bf16_sign':{f:(val>0)-(val<0) for f,val in ex.items()}});effect[name]=v
 rows.append({'id':r['id'],'split':r['split'],'group':r['group'],'natural_no_top_tie':r['natural_no_top_tie'],'gold_index':g,'fixedwrong_index':fixed,'targets':r['targets'],'conditions':cs,'contrasts':effect})
assert len(rows)==64;sets={'all':rows,'rescue':[r for r in rows if r['group']=='rescue'],'damage':[r for r in rows if r['group']=='damage'],'stable_correct':[r for r in rows if r['group']=='stable_correct'],'stable_wrong':[r for r in rows if r['group']=='stable_wrong'],'natural_no_top_tie':[r for r in rows if r['natural_no_top_tie']]};out={}
for label,rr in sets.items():
 out[label]={'n':len(rr),'conditions':{},'contrasts':{}}
 # All questions reported first; damage subset is a labeled descriptive crossing, never an inclusion gate.
 cross=collections.Counter();damaged=collections.Counter();control_cross=collections.Counter()
 for r in rr:
  c=r['conditions'];key='C10_correct='+str(c['C10']['correct'])+';C01_correct='+str(c['C01']['correct']);cross[key]+=1
  if c['C00']['correct'] and not c['C11']['correct']:damaged[key]+=1
  for ctl in ('full_norm','full_unrelated'):
   if c[ctl]['status']=='OK':control_cross[ctl+';C11_damage='+str(c['C00']['correct'] and not c['C11']['correct'])+';control_damage='+str(c['C00']['correct'] and not c[ctl]['correct'])]+=1
 out[label]['decision_cross']={'all_C10_C01_correctness':dict(cross),'C00_correct_C11_wrong_subset_C10_C01_correctness':dict(damaged),'C11_and_full_control_damage_cross':dict(control_cross)}
 for c in conditions:
  z=[r['conditions'][c] for r in rr if r['conditions'][c]['status']=='OK'];out[label]['conditions'][c]={'n':len(z),'undefined':len(rr)-len(z),'correct':sum(x['correct'] for x in z),'became_wrong':sum(x['became_wrong'] for x in z),'became_correct':sum(x['became_correct'] for x in z),'top_ties':sum(len(x['ties'])>1 for x in z),'means':{f:statistics.mean(x[f] for x in z) if z else None for f in ('allwrong','fixedwrong','dynamic')}}
 for c in contrasts:
  z=[(r['id'],r['contrasts'][c]) for r in rr if r['contrasts'][c]['status']=='OK'];item={'n':len(z),'undefined':len(rr)-len(z)}
  for field in ('allwrong','fixedwrong','dynamic'):
   v=[x[field] for _,x in z];item[field]={'mean':statistics.mean(v) if v else None,'median':statistics.median(v) if v else None,'sign_ids':{str(sign):[i for i,x in z if (x[field]>0)-(x[field]<0)==sign] for sign in (-1,0,1)}}
  item['exact_bf16_sign_ids']={field:{str(sign):[i for i,x in z if x['exact_bf16_sign'][field]==sign] for sign in (-1,0,1)} for field in ('allwrong','fixedwrong','dynamic')};out[label]['contrasts'][c]=item
OUT.mkdir(exist_ok=False)
with (OUT/'per_question.jsonl').open('x') as f:
 for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
result={'scope':'Same frozen train discovery64 (four balanced natural groups), mixed local x/a states; exploratory local causal evidence, not full natural mediation. Contrast directions exactly as Astra order. Test previously exposed; no holdout claim.','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_pending_sha256':sha(P/'PENDING.json'),'source_cpu_audit_sha256':sha(P/'CONTROL_MATH_AUDIT.json'),'source_cuda_replay_sha256':sha(P/'CUDA_REPLAY.json'),'per_question_sha256':sha(OUT/'per_question.jsonl'),'groups':out};(OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('E2_4_ANALYSIS_SHA',sha(OUT/'summary.json'))
