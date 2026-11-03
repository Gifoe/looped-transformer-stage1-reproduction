"""Classify discovery transitions; finite paths are diagnostics, not learned controllers."""
import argparse,csv,hashlib,json
import numpy as np
import torch
from runtime import ROOT,save

def table(p,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--study',default='discovery_audit');a=ap.parse_args();out=ROOT/'results'/a.study
 records=[json.loads(l) for l in (out/'scores.jsonl').read_text().splitlines()]
 assert (out/f'COMPLETE_{len(records)}.json').exists(),'Incomplete cohort; no final analysis'
 eps=.001;failures=[];geometry=[];classify=[]
 for r in records:
  m=r['methods'];p4=m['native_k4'];p5=m['native_k5']
  interior=[m[f'normalized_chord_{v}'] for v in ['0.125','0.25','0.5']]
  first=m['normalized_chord_0.125'];best=max(interior,key=lambda v:v['margin'])
  harm=p4['correct'] and not p5['correct'];rescue=not p4['correct'] and p5['correct']
  group='harm' if harm else 'rescue' if rescue else 'stable_correct' if p5['correct'] else 'stable_wrong'
  robust=bool(harm and p4['margin']>eps and p5['margin']<-eps)
  h1=bool(p5['margin']<p4['margin']-eps and first['margin']>p4['margin']+eps and best['margin']>max(p4['margin'],p5['margin'])+eps)
  classify.append(dict(id=r['id'],group=group,robust_harm=robust,h1_sampled_path=h1,
    chord_recovers_correct=bool(harm and any(x['correct'] for x in interior)),margin4=p4['margin'],margin5=p5['margin'],
    best_interior_margin=best['margin'],small_step_margin=first['margin'],
    category='H1_compatible_sampled_finite_step_failure' if h1 else 'unresolved_computation_or_readout' if harm else group))
  for before,after in [(3,4),(4,5),(5,6)]:
   b,c=m[f'native_k{before}'],m[f'native_k{after}']
   row=dict(id=r['id'],split='official_train_discovery',k_before=before,k_after=after,gold=r['gold'],
     prediction_before=b['prediction'],prediction_after=c['prediction'],correct_before=b['correct'],correct_after=c['correct'],
     margin_before=b['margin'],margin_after=c['margin'],delta_margin=c['margin']-b['margin'],
     transition='C_to_W' if b['correct'] and not c['correct'] else 'W_to_C' if c['correct'] and not b['correct'] else 'C_to_C' if c['correct'] else 'W_to_W',
     category=classify[-1]['category'] if before==4 else 'native_transition_only',epsilon=eps)
   failures.append(row)
  st=torch.load(out/(r['id'].split('/')[-1]+'_states.pt'),map_location='cpu',weights_only=True)
  for cand in st['candidates']:
   for before,after in [(3,4),(4,5),(5,6)]:
    x,y=cand['normalized'][before-1].float(),cand['normalized'][after-1].float();delta=y-x
    geometry.append(dict(id=r['id'],candidate=cand['candidate'],k_before=before,k_after=after,l2=float(delta.norm()),relative_l2=float(delta.norm()/x.norm().clamp_min(1e-12)),cosine=float(torch.nn.functional.cosine_similarity(x,y,dim=-1).mean())))
 groups={g:[c for c in classify if c['group']==g] for g in ['harm','rescue','stable_correct','stable_wrong']}
 chosen=[]
 for g,items in groups.items():
  ordered=sorted(items,key=lambda r:hashlib.sha256(('representative-v1|'+r['id']).encode()).hexdigest())
  if g=='harm':ordered=sorted(ordered,key=lambda r:not r['robust_harm'])
  chosen += [dict(**r,selection='2 per class, SHA order; robust harms prioritized before intervention outputs') for r in ordered[:2]]
 baseline=[]
 for name in records[0]['methods']:
  baseline.append(dict(method=name,n=len(records),correct=sum(r['methods'][name]['correct'] for r in records),accuracy=np.mean([r['methods'][name]['correct'] for r in records]),mean_margin=np.mean([r['methods'][name]['margin'] for r in records])))
 result=dict(status='complete',n=len(records),epsilon=eps,group_counts={g:len(v) for g,v in groups.items()},
   robust_4to5_harms=sum(c['robust_harm'] for c in classify),h1_sampled_path_all=sum(c['h1_sampled_path'] for c in classify),
   h1_sampled_path_among_harms=sum(c['h1_sampled_path'] for c in classify if c['group']=='harm'),
   contraction_recovers_harms=sum(c['chord_recovers_correct'] for c in classify),
   chosen_representatives=chosen,baseline=baseline,
   interpretation='H1 labels mean sampled finite-path compatibility, not local derivative proof. H2 and H3 are not assigned without causal module/readout controls. Per-case best coefficient is diagnostic oracle only.')
 save(str(out.relative_to(ROOT)/'ANALYSIS.json'),result);table(out/'failure_mechanisms.csv',failures);table(out/'hidden_geometry.csv',geometry);table(out/'depth_and_readout_summary.csv',baseline)
 save('configs/representative_cases.json',dict(source_study=a.study,source_n=len(records),cases=chosen))
 print(json.dumps({k:result[k] for k in ['n','group_counts','robust_4to5_harms','h1_sampled_path_among_harms','contraction_recovers_harms']},indent=2))
if __name__=='__main__':main()
