import json,gzip,pathlib,hashlib,collections,numpy as np,datetime
import torch
from fractions import Fraction
from safetensors.torch import load_file
R=pathlib.Path('/home/luansiyu/loop-latent');B=R/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E2';P=E/'state_intervention/discovery64';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((P/'CUDA_REPLAY_64_STRICT.json').read_text())['status']=='CUDA_INDEPENDENT_SAVED_STATE_LOGITS_REPLAY_PASSED';assert json.loads((P/'CONTROL_MATH_AUDIT.json').read_text())['status']=='E2_DISCOVERY64_INDEPENDENT_CONTROL_MATH_PASSED';pending=json.loads((P/'PENDING_INDEPENDENT_AUDIT.json').read_text());assert len(pending['cases'])==64
with gzip.open(E/'E2_latent_transition_rows.jsonl.gz','rt') as f:rows={(r['split'],r['id']):r for r in map(json.loads,f)}
Q=[];expected=('identity','rollback','rollback_complement','rollback_full','rollback_offered','norm_matched','unrelated','random_rank4','restore','forward_add')
def metrics(scores,g,fixed):
 wrong=[i for i in range(len(scores)) if i!=g];top=max(scores);ties=[i for i,v in enumerate(scores) if v==top];return {'candidate_scores':scores,'prediction':ties[0],'top_ties':ties,'correct':ties[0]==g,'gold_allwrong':scores[g]-float(np.mean([scores[i] for i in wrong])),'fixedwrong':scores[g]-scores[fixed],'dynamic':scores[g]-max(scores[i] for i in wrong)}
for c in pending['cases']:
 k=hashlib.sha256((c['split']+'\0'+c['id']).encode()).hexdigest();p=P/(k+'.json');assert sha(p)==c['metadata_sha256'];m=json.loads(p.read_text());T=load_file(str(P/(k+'.safetensors')));r=rows[(c['split'],c['id'])];assert r['targets']==m['targets'];g=r['gold'];wrong=[i for i in range(len(m['targets'])) if i!=g];fixed=max(wrong,key=lambda i:r['native_scores'][0][i]);native={str(u):metrics(r['native_scores'][u-1],g,fixed) for u in (1,2)};conditions={}
 for name in expected:
  d=m['conditions'][name]
  if d['status']!='OK':conditions[name]={'status':'UNDEFINED'};continue
  t=metrics(d['candidate_full_vocab_logprob'],g,fixed);t['candidate_logits_bf16']=d['candidate_logits'];base=native['1' if name=='forward_add' else '2'];t.update({'status':'OK','delta_gold_allwrong':t['gold_allwrong']-base['gold_allwrong'],'delta_fixedwrong':t['fixedwrong']-base['fixedwrong'],'delta_dynamic':t['dynamic']-base['dynamic'],'became_wrong':base['correct'] and not t['correct'],'became_correct':not base['correct'] and t['correct']});conditions[name]=t
 for name in expected:
  if conditions[name]['status']!='OK' or name=='forward_add':continue
  x=conditions[name]['candidate_logits_bf16'];y=conditions['identity']['candidate_logits_bf16'];exact=lambda vals:Fraction.from_float(vals[g])-sum((Fraction.from_float(vals[i]) for i in wrong),Fraction())/len(wrong);change=exact(x)-exact(y);conditions[name]['exact_bf16_logit_allwrong_delta_sign']=(change>0)-(change<0);conditions[name]['exact_bf16_logit_allwrong_delta']=str(change)
  stages={}
  for stage in ('Q_R','A_R_fp64','sdpa_R','a_R'):
   ref=T['identity_'+stage].double().reshape(-1);val=T[name+'_'+stage].double().reshape(-1);dn=(val-ref).norm();den=ref.norm()*val.norm();stages[stage]={'delta_l2':float(dn),'cosine':float(torch.dot(ref,val)/den) if den>0 else None}
  conditions[name]['stage_vs_identity_x2']=stages
 Q.append({'id':r['id'],'split':r['split'],'gold_index':g,'targets':r['targets'],'fixedwrong_index':fixed,'group':r['group'],'natural_no_top_tie':r['natural_no_top_tie'],'native':native,'offered_span':{**m['offered_span'],'complement_delta_l2':float((T['delta_fp64']-T['component_fp64']).norm())},'conditions':conditions})
def agg(rs,name):
 valid=[r['conditions'][name] for r in rs if r['conditions'][name]['status']=='OK'];ret={'n':len(rs),'defined':len(valid),'undefined':len(rs)-len(valid)}
 if not valid:return ret
 for field in ('gold_allwrong','fixedwrong','dynamic','delta_gold_allwrong','delta_fixedwrong','delta_dynamic'):ret['mean_'+field]=float(np.mean([x[field] for x in valid]))
 for field in ('correct','became_wrong','became_correct'):ret[field+'_count']=sum(x[field] for x in valid)
 ret['top_ties']=sum(len(x['top_ties'])>1 for x in valid);
 if 'exact_bf16_logit_allwrong_delta_sign' in valid[0]:ret['exact_bf16_logit_allwrong_delta_sign_counts']={str(sign):sum(x['exact_bf16_logit_allwrong_delta_sign']==sign for x in valid) for sign in (-1,0,1)}
 return ret
sets={'all':Q,'rescue':[q for q in Q if q['group']=='rescue'],'damage':[q for q in Q if q['group']=='damage'],'stable_correct':[q for q in Q if q['group']=='stable_correct'],'stable_wrong':[q for q in Q if q['group']=='stable_wrong'],'natural_no_top_tie':[q for q in Q if q['natural_no_top_tie']]};summary={k:{n:agg(v,n) for n in expected} for k,v in sets.items()};compare={}
for control in ('norm_matched','unrelated','random_rank4','rollback_complement','rollback_full','rollback_offered'):
 compare[control]={}
 for name,rs in sets.items():
  a=[(r['conditions']['rollback'],r['conditions'][control]) for r in rs if r['conditions']['rollback']['status']==r['conditions'][control]['status']=='OK'];compare[control][name]={'n':len(a),'mean_P_minus_control_gold_allwrong':float(np.mean([x['gold_allwrong']-y['gold_allwrong'] for x,y in a])) if a else None,'mean_P_minus_control_fixedwrong':float(np.mean([x['fixedwrong']-y['fixedwrong'] for x,y in a])) if a else None,'mean_P_minus_control_dynamic':float(np.mean([x['dynamic']-y['dynamic'] for x,y in a])) if a else None,'P_wrong_control_correct':sum(not x['correct'] and y['correct'] for x,y in a),'P_correct_control_wrong':sum(x['correct'] and not y['correct'] for x,y in a),'median_P_minus_control_gold_allwrong':float(np.median([x['gold_allwrong']-y['gold_allwrong'] for x,y in a])) if a else None,'sign_counts':{str(sign):sum((x['gold_allwrong']>y['gold_allwrong'])-(x['gold_allwrong']<y['gold_allwrong'])==sign for x,y in a) for sign in (-1,0,1)}}
energy={k:{'n':len(rs),'mean_complement_delta_l2':float(np.mean([r['offered_span']['complement_delta_l2'] for r in rs])),'mean_global_delta_l2':float(np.mean([r['offered_span']['global_delta_l2'] for r in rs])),'mean_offered_delta_l2':float(np.mean([r['offered_span']['offered_delta_l2'] for r in rs])),'mean_global_minus_offered_delta_l2':float(np.mean([r['offered_span']['global_minus_offered_delta_l2'] for r in rs])),'offered_rank_counts':dict(collections.Counter(str(r['offered_span']['rank']) for r in rs))} for k,rs in sets.items() if rs}
out=E/'state_intervention/discovery64_analysis';out.mkdir(exist_ok=False)
with (out/'per_question.jsonl').open('x') as f:
 for q in Q:f.write(json.dumps(q,allow_nan=False)+'\n')
result={'scope':'Pre-frozen 64 train discovery, stratified 16 each: exploratory, not unbiased accuracy or holdout; R-only L18 residual perturbation changes Q/K/V and residual jointly. Test effects were not used to select this sample. Fixedwrong is K1 native best wrong. No group selection or re-fitting.','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage_scope':'Relative to same-input identity-x2 controlled forward; endpoint full logits match no-hook native K2 bitwise, but native intermediate stages were not independently saved or compared. Descriptive only.', 'source_pending_sha256':sha(P/'PENDING_INDEPENDENT_AUDIT.json'),'source_cpu_audit_sha256':sha(P/'CONTROL_MATH_AUDIT.json'),'source_cuda_replay_sha256':sha(P/'CUDA_REPLAY_64_STRICT.json'),'per_question_sha256':sha(out/'per_question.jsonl'),'summaries':summary,'P_vs_controls':compare,'energy':energy}
(out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('E2_DISCOVERY64_ANALYSIS',sha(out/'summary.json'),'per_question',result['per_question_sha256'])
for n in expected: print(n,summary['all'][n])
