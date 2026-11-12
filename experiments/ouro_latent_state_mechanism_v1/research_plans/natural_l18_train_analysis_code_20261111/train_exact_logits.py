"""Exact rational BF16 target-logit sensitivity; production score metrics unchanged."""
import json,gzip,hashlib,os
from fractions import Fraction
from pathlib import Path
ROOT=Path('/home/luansiyu/loop-latent/research_plans');BASE=ROOT/'natural_l18_train_analysis_20261111';COV=ROOT/'NATURAL_L18_TRAIN_PRELIMINARY_COVERAGE_20261111.json';OUT=ROOT/'natural_l18_train_exact_logits_20261111'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def sign(x):return (x>0)-(x<0)
def main():
 rf=BASE/'rows.jsonl.gz';require=sha(rf)=='c5270ee6798f5893914c936c009469c5a89ea25fb486dfef7c9c4fbc8aea91e3' and sha(COV)=='32185c796d722f937c93845d46a4b8d9fb7a4066f5892b82ce12f2a9e9d3d2e7'
 if not require:raise ValueError('source SHA')
 with gzip.open(rf,'rt') as f:rows=[json.loads(x) for x in f]
 if len(rows)!=1119 or len({r['id'] for r in rows})!=1119:raise ValueError('IDs')
 result=[]
 for r in rows:
  gold=r['gold'];wrong=r['fixed_wrong'];other=[j for j in range(len(r['targets'])) if j!=gold];raw={k:[Fraction.from_float(x) for x in v] for k,v in r['target_logits_bf16'].items()}
  def effect(a,b,ref):
   ids=other if ref=='all_wrong' else [wrong];return sum(((raw[a][gold]-raw[a][j])-(raw[b][gold]-raw[b][j]) for j in ids),Fraction(0))/len(ids)
  v={}
  for name,a,b in [('L2','B2','T21'),('G1','T12','B1')]:
   v[name]={}
   for ref in ('all_wrong','fixed_wrong'):
    z=effect(a,b,ref);production=r[name][ref];v[name][ref]={'production':production,'raw_numerator':z.numerator,'raw_denominator':z.denominator,'raw_value':float(z),'production_sign':sign(production),'raw_sign':sign(z)}
  result.append({'id':r['id'],'demo':r['demo'],'group':r['group'],'natural_no_top_tie':r['natural_no_top_tie'],'metrics':v})
 OUT.mkdir(exist_ok=False);of=OUT/'rows.jsonl.gz'
 with gzip.open(of,'xt') as f:
  for r in result:f.write(json.dumps(r,sort_keys=True)+'\n')
 groups={'all':result,'non_demo':[r for r in result if not r['demo']],'demo':[r for r in result if r['demo']]}
 for scope in ('all','non_demo'):
  base=groups[scope];groups[scope+'/natural_no_tie']=[r for r in base if r['natural_no_top_tie']]
  for group in ('rescue','damage','stable_correct','stable_wrong'):groups[scope+'/'+group]=[r for r in base if r['group']==group]
 summary={}
 for group,rr in groups.items():
  d={}
  for name in ('L2','G1'):
   for ref in ('all_wrong','fixed_wrong'):
    items=[r['metrics'][name][ref] for r in rr];d[name+'_'+ref]={'n':len(items),'production_positive':sum(x['production_sign']>0 for x in items),'production_negative':sum(x['production_sign']<0 for x in items),'production_zero':sum(x['production_sign']==0 for x in items),'raw_positive':sum(x['raw_sign']>0 for x in items),'raw_negative':sum(x['raw_sign']<0 for x in items),'raw_zero':sum(x['raw_sign']==0 for x in items),'sign_conflicts':sum(x['production_sign']!=x['raw_sign'] for x in items),'production_nonzero_raw_zero':sum(x['production_sign']!=0 and x['raw_sign']==0 for x in items),'strict_opposite':sum(x['production_sign']*x['raw_sign']<0 for x in items)}
  summary[group]=d
 out={'status':'TRAIN_EXACT_BF16_TARGET_LOGIT_SIGN_SENSITIVITY_CUDA_REPLAY_PENDING','source_rows_sha256':sha(rf),'source_coverage_sha256':sha(COV),'rows_sha256':sha(of),'script_sha256':sha(Path(__file__)),'semantics':'exact rational differences of candidate BF16 logits; not original production scores or a replacement outcome','groups':summary}
 with (OUT/'summary.json').open('x') as f:json.dump(out,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 print('TRAIN_EXACT_LOGITS_COMPLETE',len(result),flush=True)
if __name__=='__main__':main()
