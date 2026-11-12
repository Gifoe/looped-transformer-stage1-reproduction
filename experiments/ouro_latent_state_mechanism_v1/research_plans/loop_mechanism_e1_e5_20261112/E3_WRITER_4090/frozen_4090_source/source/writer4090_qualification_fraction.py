import json,pathlib,hashlib,statistics
from fractions import Fraction
R=pathlib.Path('/root/autodl-tmp/ouro-e1-e3-offload-20261113/results/analysis_train64');p=R/'rows.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];summary=json.loads((R/'summary.json').read_text());out={}
for b in ('a','b'):
 out[b]={}
 for name in ('rest','loss','rest_norm','loss_norm','rest_specific','loss_specific'):
  out[b][name]={}
  for key,pred in [('rescue',lambda r:r['actual_group_4090']=='rescue'),('rescue_no_tie',lambda r:r['actual_group_4090']=='rescue' and r['natural_no_tie'])]:
   rr=[r for r in rows if pred(r) and r['contrasts'][b][name] is not None];prod=[r['contrasts'][b][name]['allwrong'] for r in rr];raw=[Fraction(r['raw_bf16_fraction_contrasts'][b][name]['allwrong']) for r in rr];mean=sum(raw,Fraction())/len(raw) if raw else None;pm=statistics.mean(prod) if prod else None;out[b][name][key]={'n':len(rr),'production_mean':pm,'raw_fraction_mean':str(mean) if mean is not None else None,'raw_mean_float':float(mean) if mean is not None else None,'raw_sign':(1 if mean>0 else -1 if mean<0 else 0) if mean is not None else None,'production_sign':(1 if pm>0 else -1 if pm<0 else 0) if pm is not None else None,'ambiguous_sign':(pm>0 and mean<=0) if mean is not None else None}
q=R/'qualification_fraction_sensitivity.json';assert not q.exists();q.write_text(json.dumps({'rows_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'summary_sha256':hashlib.sha256((R/'summary.json').read_bytes()).hexdigest(),'branches':out},indent=2)+'\n');print(json.dumps(out,indent=2))
