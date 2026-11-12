"""Per-question descriptive state changes; not mechanistic or causal evidence."""
import argparse,hashlib,json,os
from pathlib import Path
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from ledger_guard import require_production_ledger
from cohort_runner import PROJECT,split_data_path,bound_hashes,sha

def pair_stats(x,y):
 x=x.float();y=y.float()
 if x.ndim!=2 or x.shape!=y.shape or not len(x):raise ValueError('bad state shapes')
 if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('nonfinite states')
 nx=x.norm(dim=-1);ny=y.norm(dim=-1);valid=(nx>0)&(ny>0);cos=((x[valid]*y[valid]).sum(-1)/(nx[valid]*ny[valid])).mean().item() if bool(valid.any()) else None
 return {'tokens':len(x),'mean_norm_before':nx.mean().item(),'mean_norm_after':ny.mean().item(),'mean_l2_change':(y-x).norm(dim=-1).mean().item(),'mean_cosine':cos,'cosine_valid_tokens':int(valid.sum())}

def main():
 if not __debug__:raise RuntimeError('assertions required; refuse python -O')
 ap=argparse.ArgumentParser();ap.add_argument('--split',required=True,choices=['train','validation','test']);ap.add_argument('--shards',required=True);ap.add_argument('--coverage',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 shards=Path(a.shards).resolve();shards.relative_to(PROJECT.resolve());out=Path(a.out).resolve();out.relative_to(PROJECT.resolve());cov=json.loads(Path(a.coverage).read_text());require_production_ledger(cov);counts=cov['counts'];rows=pq.read_table(split_data_path(a.split)).to_pylist()
 if counts['complete_questions']!=len(rows) or counts['expected_questions']!=len(rows) or counts['complete_requests']!=counts['expected_requests'] or counts['failed_questions'] or counts['missing_questions']:raise RuntimeError('incomplete coverage')
 ledger={x['id']:x for x in cov['questions']}
 if len(ledger)!=len(rows) or set(ledger)!={r['id'] for r in rows} or any(x['status']!='complete' or x['split']!=a.split for x in ledger.values()):raise RuntimeError('invalid coverage ledger')
 expected=bound_hashes(a.split);tmp=out.with_suffix(out.suffix+'.partial')
 with tmp.open('w',encoding='utf8') as f:
  for row in rows:
   key=hashlib.sha256(row['id'].encode()).hexdigest();meta=json.loads((shards/(key+'.json')).read_text());path=shards/(key+'.safetensors')
   if meta['id']!=row['id'] or meta['split']!=a.split or meta['sha256']!=expected or not path.is_file() or sha(path)!=meta['tensor_sha256'] or ledger[row['id']]['tensor_sha256']!=meta['tensor_sha256'] or ledger[row['id']]['metadata_sha256']!=sha(shards/(key+'.json')):raise RuntimeError('stale or corrupt shard')
   records=meta['records'];n=len(row['choices']['text'])
   if len(records)!=n or [r['candidate'] for r in records]!=list(range(n)) or any(len(r['scores'])!=4 or not all(__import__('math').isfinite(z) for z in r['scores']) for r in records):raise RuntimeError('invalid candidate coverage or scores')
   first=records[0];enc=first['encoding'];
   if any(r['encoding']['input_ids']!=enc['input_ids'] or r['captured_positions']!=first['captured_positions'] for r in records):raise RuntimeError('candidate inputs differ; cannot use candidate 0 as common representation')
   positions=first['captured_positions'];lookup={p:i for i,p in enumerate(positions)};roles={name:[lookup[p] for p in pos] for name,pos in enc['spans'].items()};roles['readout']=[lookup[p] for p in enc['predecessor_positions']]
   result={'id':row['id'],'split':a.split,'candidate_for_common_input':0,'input_ids_sha256':first['input_ids_sha256'],'prediction_trajectory':meta['predictions'],'gold_index':meta['gold_index'],'observations':[]}
   with safe_open(str(path),framework='pt',device='cpu') as sf:
    for site in [f'layer{i:02d}' for i in range(24)]+['post_norm']:
     states=[sf.get_tensor(f'candidate0_loop{k}_{site}') for k in range(1,5)]
     for loop in range(3):
      for role,indices in roles.items():
       result['observations'].append({'site':site,'transition':[loop+1,loop+2],'role':role,**pair_stats(states[loop][indices],states[loop+1][indices])})
   f.write(json.dumps(result,allow_nan=False)+'\n')
 os.link(tmp,out,follow_symlinks=False);tmp.unlink()
 receipt={'split':a.split,'question_count':len(rows),'output_sha256':sha(out),'coverage_sha256':sha(Path(a.coverage)),'script_sha256':sha(Path(__file__)),'data_sha256':sha(split_data_path(a.split)),'scope':'descriptive state changes; no causal claim'}
 from cohort_runner import write_new_json
 write_new_json(Path(str(out)+'.receipt.json'),receipt)
if __name__=='__main__':main()
