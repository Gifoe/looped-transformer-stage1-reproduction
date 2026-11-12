import json,pathlib,hashlib,collections,statistics
from fractions import Fraction
import numpy as np,torch
from safetensors.torch import load_file
B=pathlib.Path('/home/luansiyu/loop-latent/repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112');P=B/'E2/state_intervention/discovery64';A=B/'E2/state_intervention/discovery64_analysis';rows=[json.loads(x) for x in (A/'per_question.jsonl').open()];assert len(rows)==64
sets={'all':rows,'rescue':[r for r in rows if r['group']=='rescue'],'damage':[r for r in rows if r['group']=='damage'],'stable_correct':[r for r in rows if r['group']=='stable_correct'],'stable_wrong':[r for r in rows if r['group']=='stable_wrong'],'natural_no_top_tie':[r for r in rows if r['natural_no_top_tie']]};controls=('norm_matched','unrelated','random_rank4','rollback_complement','rollback_full','rollback_offered');cmp={}
for ctl in controls:
 cmp[ctl]={}
 for label,rs in sets.items():
  vals=[];fraction_sign={'-1':[],'0':[],'1':[]};production_sign={'-1':[],'0':[],'1':[]}
  for r in rs:
   a=r['conditions']['rollback'];b=r['conditions'][ctl]
   if a['status']!='OK' or b['status']!='OK':continue
   delta=a['gold_allwrong']-b['gold_allwrong'];vals.append(delta);sign=(delta>0)-(delta<0);production_sign[str(sign)].append(r['id']);g=r['gold_index'];wrong=[i for i in range(len(r['targets'])) if i!=g];exact=lambda z:Fraction.from_float(z[g])-sum((Fraction.from_float(z[i]) for i in wrong),Fraction())/len(wrong);fx=exact(a['candidate_logits_bf16'])-exact(b['candidate_logits_bf16']);fraction_sign[str((fx>0)-(fx<0))].append(r['id'])
  cmp[ctl][label]={'N':len(vals),'mean_P_minus_control':statistics.mean(vals) if vals else None,'median_P_minus_control':statistics.median(vals) if vals else None,'production_sign_ids':production_sign,'exact_bf16_fraction_sign_ids':fraction_sign}
for r in rows:
 key=hashlib.sha256((r['split']+'\0'+r['id']).encode()).hexdigest();T=load_file(str(P/(key+'.safetensors')));x2=T['identity_original_R'].double();x1=T['forward_add_original_R'].double();r['intervention_norms']={}
 for name,c in r['conditions'].items():
  if c['status']!='OK':continue
  target=x1 if name=='forward_add' else x2;cast=T.get(name+'_cast_residual_fp64');r['intervention_norms'][name]={'receiver_norm':float(target.norm()),'injected_norm':float(T[name+'_injected_R'].double().norm()),'actual_removed_or_added_norm':float((T[name+'_injected_R'].double()-target).norm()),'cast_residual_norm':float(cast.norm()) if cast is not None else 0.}
stages={}
for group,rs in sets.items():
 stages[group]={}
 for name in ('rollback','rollback_complement','rollback_full','rollback_offered','norm_matched','unrelated','random_rank4','identity','restore'):
  stages[group][name]={}
  for stage in ('Q_R','A_R_fp64','sdpa_R','a_R'):
   v=[r['conditions'][name]['stage_vs_identity_x2'][stage] for r in rs if r['conditions'][name]['status']=='OK'];cs=[x['cosine'] for x in v if x['cosine'] is not None];stages[group][name][stage]={'n':len(v),'mean_delta_l2':float(np.mean([x['delta_l2'] for x in v])) if v else None,'mean_cosine':float(np.mean(cs)) if cs else None}
res={'scope':'Paired comparisons and exact BF16-logit fractional sign checks from frozen 64; stage deltas relative to same-input identity-x2. Terminal logits identity matches nativeK2 bitwise, native intermediate stages were not saved or independently verified. Exploratory, not mediator identification.','source_per_question_sha256':hashlib.sha256((A/'per_question.jsonl').read_bytes()).hexdigest(),'paired':cmp,'stages':stages,'per_question_norms':{r['id']:r['intervention_norms'] for r in rows}};out=A/'paired_stage_norm_supplement.json';assert not out.exists();out.write_text(json.dumps(res,indent=2,allow_nan=False)+'\n');print('SUPPLEMENT_SHA',hashlib.sha256(out.read_bytes()).hexdigest());
for ctl in controls:
 x=cmp[ctl]['all'];print(ctl,'mean',x['mean_P_minus_control'],'median',x['median_P_minus_control'],'production sign counts',{k:len(v) for k,v in x['production_sign_ids'].items()},'exact sign counts',{k:len(v) for k,v in x['exact_bf16_fraction_sign_ids'].items()})
