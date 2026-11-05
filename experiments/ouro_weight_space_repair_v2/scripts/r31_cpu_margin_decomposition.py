"""Postrun all24 exact arithmetic attribution, no model/search/new scoring method."""
import json,hashlib
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def winner(v):return max(range(len(v)),key=lambda i:v[i])
def sign(v):return (v>0)-(v<0)
def run(tag):
 s=read(f'results/{tag}_SCORES.json');a=read(f'results/{tag}_ANALYSIS.json');c=read(f'results/{tag}_COMPLETE.json')
 assert c['status']=='complete' and a['status']=='complete_analyzed_once' and c['scores_sha256']==sha(f'results/{tag}_SCORES.json')
 preds={r['id']:r for r in a['predictions']};records=[]
 for row in s['rows']:
  cs=row['candidates'];y=preds[row['id']]['gold'];p=[];f=[];l=[];p_rat=[];f_rat=[];l_rat=[]
  for q in cs:
   chars=q['characters'];p.append(q['pieces']['primary']['sum_score']/chars);f.append(q['pieces']['Fixed']['sum_score']/chars);l.append(q['late'])
   assert p[-1]+l[-1]==q['scores']['primary'] and f[-1]+l[-1]==q['scores']['A1']
   p_rat.append(F(q['pieces']['primary']['sum_score'])/chars);f_rat.append(F(q['pieces']['Fixed']['sum_score'])/chars)
   l_rat.append(F(q['pieces']['native4']['sum_score'])/chars-F(q['pieces']['native5']['sum_score'])/chars)
  post=[q['scores']['primary'] for q in cs];base=[q['scores']['A1'] for q in cs]
  assert winner(post)==preds[row['id']]['predictions']['primary'] and winner(base)==preds[row['id']]['predictions']['A1']
  pairs=[]
  for j in range(len(cs)):
   if j==y:continue
   pm=p_rat[y]-p_rat[j];fm=f_rat[y]-f_rat[j];lm=l_rat[y]-l_rat[j];delta=pm-fm
   pairs.append(dict(wrong=j,pre_margin=float(pm),late_margin=float(lm),post_components_margin=float(pm+lm),correction_margin=float(delta),exact_signs=dict(pre=sign(pm),late=sign(lm),correction=sign(delta)),exact_ratios={k:str(v) for k,v in [('pre',pm),('late',lm),('correction',delta)]}))
  fp=winner(f);pp=winner(p);po=winner(post)
  records.append(dict(id=row['id'],gold=y,R6_wrong=preds[row['id']]['predictions']['R6_historical']!=y,Fixed_pre_prediction=fp,primary_pre_prediction=pp,primary_post_prediction=po,late_veto=pp==y and po!=y,late_rescue=pp!=y and po==y,pre_correction_rescue=fp!=y and pp==y,pre_correction_damage=fp==y and pp!=y,pairs=pairs))
 assert len(records)==24 and len(preds)==24
 def counts(rs):return dict(n=len(rs),**{k:sum(r[k] for r in rs) for k in ['late_veto','late_rescue','pre_correction_rescue','pre_correction_damage']},primary_pre_correct=sum(r['primary_pre_prediction']==r['gold'] for r in rs),primary_post_correct=sum(r['primary_post_prediction']==r['gold'] for r in rs),Fixed_pre_correct=sum(r['Fixed_pre_prediction']==r['gold'] for r in rs))
 return dict(all24=counts(records),R6_errors=counts([r for r in records if r['R6_wrong']]),records=records)
def main():
 out=ROOT/'results/r31_CPU_MARGIN_DECOMPOSITION.json';assert not out.exists()
 tags=['r28','r30'];inputs=['r31_CPU_MARGIN_DIAGNOSTIC_PROTOCOL.md','scripts/r31_cpu_margin_decomposition.py']+[f'results/{t}_{x}.json' for t in tags for x in ['SCORES','ANALYSIS','COMPLETE']]
 result=dict(status='complete_postrun_CPU_only',scope='All24 previously unsealed; no coefficient search, model, heldout or deployable no-late proposal',input_sha256={p:sha(p) for p in inputs},methods={t:run(t) for t in tags})
 out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({t:v['all24'] for t,v in result['methods'].items()}))
if __name__=='__main__':main()
