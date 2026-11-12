"""Scalar-only reanalysis from recovered complete JSON; no model or activation tensors needed."""
import argparse,gzip,json,csv
from pathlib import Path
from fractions import Fraction
a=argparse.ArgumentParser();a.add_argument('--recovered',required=True);a.add_argument('--labels',required=True);a.add_argument('--output',required=True);args=a.parse_args();op=gzip.open if args.labels.endswith('.gz') else open
with op(args.labels,'rt',encoding='utf8') as f:labels={(q['split'],q['id']):q['gold'] for q in map(json.loads,f)}
def am(v):return max(range(len(v)),key=lambda i:v[i])
def metrics(v,g,f,globalf):return {'meanwrong':v[g]-sum(x for i,x in enumerate(v) if i!=g)/(len(v)-1),'fixedwrong':v[g]-v[f],'dynamicmargin':v[g]-max(x for i,x in enumerate(v) if i!=g),'global_K1_fixedwrong':v[g]-v[globalf]}
rows=[]
for p in sorted(Path(args.recovered).rglob('*.results.json')):
 obj=json.loads(p.read_text());m=json.loads(p.with_name(p.name.replace('.results.json','.native.json')).read_text());g=labels[(obj['split'],obj['id'])];first=m['native'].get('1',m.get('reused_K1_native'))
 for r in obj['rows']:
  d=str(r['receiver_loop']);base=m['native'][d];v=r['scores'];b=base['scores'];fixed=max((i for i in range(len(b)) if i!=g),key=lambda i:b[i]);globalf=max((i for i in range(len(b)) if i!=g),key=lambda i:first['scores'][i]);iv=metrics(v,g,fixed,globalf);bv=metrics(b,g,fixed,globalf);exact=metrics(list(map(Fraction,r['candidate_logits'])),g,fixed,globalf);be=metrics(list(map(Fraction,base['candidate_logits'])),g,fixed,globalf)
  row={'run':p.parent.name,'split':obj['split'],'id':obj['id'],'condition':r['condition'],'receiver_loop':r['receiver_loop'],'correct':int(am(v)==g),'rescue':int(am(b)!=g and am(v)==g),'damage':int(am(b)==g and am(v)!=g),'top_ties':sum(x==max(v) for x in v)>1}
  for k in iv:row[k]=iv[k];row[k+'_delta']=iv[k]-bv[k];row[k+'_exact_raw_delta_sign']=int(exact[k]>be[k])-int(exact[k]<be[k])
  rows.append(row)
assert rows
with open(args.output,'w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'status':'SCALAR_RECOMPUTATION_COMPLETE','rows':len(rows),'output':args.output}))
