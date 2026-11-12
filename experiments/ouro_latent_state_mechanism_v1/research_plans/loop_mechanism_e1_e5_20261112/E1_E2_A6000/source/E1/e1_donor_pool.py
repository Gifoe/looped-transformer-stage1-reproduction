"""Frozen train-only unrelated donor selection for E1 at identical Q background."""
import json,gzip,hashlib
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path('/home/luansiyu/loop-latent');OLD=ROOT/'research_plans/natural_qkv_eight_corner_20261112';SEED='e1-unrelated-20261112'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
class Pool:
 def __init__(self):
  cov=OLD/'train_independent_cuda_coverage.json';summary=OLD/'train_analysis/summary.json';rows=OLD/'train_analysis/rows.jsonl.gz';c=json.loads(cov.read_text());s=json.loads(summary.read_text());assert sha(cov)=='d257487676f31d0d122a913caad2c85369629ce1acca21bdabb5b2de0603c381' and c['count']==1119 and s['coverage_sha256']==sha(cov) and s['rows_sha256']==sha(rows)
  with gzip.open(rows,'rt') as f:rr=[json.loads(x) for x in f]
  self.ids=sorted(r['id'] for r in rr if not r['demo']);self.ledger={q['id']:q for q in c['questions']};assert len(self.ids)==1094 and len(self.ledger)==1119;self.pool_sha=hashlib.sha256(json.dumps(self.ids).encode()).hexdigest();self.coverage_sha=sha(cov);self.rows_sha=sha(rows)
 def select(self,split,ident,input_sha,q):
  start=int(hashlib.sha256(f'{split}\0{ident}\0{q}\0{SEED}'.encode()).hexdigest(),16)%len(self.ids)
  for k in range(len(self.ids)):
   donor=self.ids[(start+k)%len(self.ids)]
   if split=='train' and ident==donor:continue
   key=hashlib.sha256(donor.encode()).hexdigest();mp=OLD/'train'/(key+'.json');tp=mp.with_suffix('.safetensors');l=self.ledger[donor]
   if sha(mp)!=l['metadata_sha256'] or sha(tp)!=l['tensor_sha256']:raise ValueError('donor hash')
   m=json.loads(mp.read_text())
   if m['input_ids_sha256']==input_sha:continue
   with safe_open(str(tp),framework='pt',device='cpu') as f:
    ss=[f.get_tensor(f'G{q}{j}{k}_sdpa_R').to(torch.float64) for j,k in ((1,1),(1,2),(2,1),(2,2))]
   add=(ss[2]+ss[1]-ss[0]).to(torch.bfloat16);d=ss[3]-add.double()
   if torch.linalg.vector_norm(d)==0:continue
   return d,{'id':donor,'split':'train','input_sha256':m['input_ids_sha256'],'metadata_sha256':l['metadata_sha256'],'tensor_sha256':l['tensor_sha256'],'pool_sha256':self.pool_sha,'coverage_sha256':self.coverage_sha,'rows_sha256':self.rows_sha,'seed':SEED,'q':q}
  raise ValueError('no valid unrelated donor')
