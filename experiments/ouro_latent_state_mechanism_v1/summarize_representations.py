"""Aggregate descriptive representation changes with equal question weighting."""
import argparse,collections,json
from pathlib import Path
import pyarrow.parquet as pq
from cohort_runner import write_new_json
from cohort_runner import PROJECT,split_data_path,sha,write_json
METRICS=('mean_norm_before','mean_norm_after','mean_l2_change','mean_cosine')
class Accumulator:
 def __init__(self):self.data={}
 def add(self,key,observation):
  bucket=self.data.setdefault(key,{m:[0.,0] for m in METRICS})
  for m in METRICS:
   x=observation[m]
   if x is not None:bucket[m][0]+=x;bucket[m][1]+=1
 def finish(self):
  result=[]
  for key,bucket in sorted(self.data.items()):
   result.append({'site':key[0],'transition':list(key[1]),'role':key[2],'outcome':key[3],'metrics':{m:{'mean':s/n if n else None,'questions':n} for m,(s,n) in bucket.items()}})
  return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',required=True,choices=['train','validation','test']);ap.add_argument('--input',required=True);ap.add_argument('--coverage',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 path=Path(a.input).resolve();out=Path(a.out).resolve();path.relative_to(PROJECT.resolve());out.relative_to(PROJECT.resolve());receipt=json.loads(Path(str(path)+'.receipt.json').read_text())
 if receipt['split']!=a.split or receipt['output_sha256']!=sha(path) or receipt['coverage_sha256']!=sha(Path(a.coverage)) or receipt['data_sha256']!=sha(split_data_path(a.split)) or receipt['script_sha256']!=sha(Path(__file__).with_name('representation_observation.py')):raise RuntimeError('stale observation artifact')
 rows=pq.read_table(split_data_path(a.split)).to_pylist();expected={r['id'] for r in rows};seen=set();acc=Accumulator()
 if len(expected)!=len(rows) or receipt['question_count']!=len(rows):raise RuntimeError('input coverage mismatch')
 with path.open(encoding='utf8') as f:
  for line in f:
   r=json.loads(line);ident=r['id'];pred=r['prediction_trajectory'];gold=r['gold_index']
   if ident not in expected or ident in seen or r['split']!=a.split or len(pred)!=4:raise RuntimeError('invalid observation record')
   seen.add(ident)
   for ob in r['observations']:
    k=ob['transition'][0]-1
    if not 0<=k<3 or ob['transition']!=[k+1,k+2]:raise RuntimeError('invalid transition')
    before=pred[k]==gold;after=pred[k+1]==gold;outcome=('stable_correct' if after else 'damage') if before else ('rescue' if after else 'stable_wrong')
    for group in ('all',outcome):acc.add((ob['site'],tuple(ob['transition']),ob['role'],group),ob)
 if seen!=expected:raise RuntimeError('missing observation record')
 write_new_json(out,{'scope':'descriptive equal-question summaries; not causal','split':a.split,'question_count':len(seen),'overlapping_roles':'tokens may be included in multiple semantic roles','groups':acc.finish()})
if __name__=='__main__':main()
