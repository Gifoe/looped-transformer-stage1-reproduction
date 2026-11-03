"""Train-only V2 expansion; preserves the exact V1 demonstrations and selection."""
import hashlib,json
import pyarrow.parquet as pq
from runtime import ROOT,save
from build_train_manifests import fingerprint,record,render,write

def main():
 old=[json.loads(l) for l in (ROOT/'data/discovery.jsonl').read_text().splitlines()]
 selection=[json.loads(l) for l in (ROOT/'data/selection.jsonl').read_text().splitlines()]
 demos=json.loads((ROOT/'data/fixed_demonstrations.json').read_text())['documents']
 assert len(old)==len(selection)==192 and len(demos)==25
 seen={fingerprint(d['question']) for d in demos};pool=[]
 for i,doc in enumerate(pq.read_table(ROOT/'data/raw/train.parquet').to_pylist()):
  key=fingerprint(doc['question'])
  if key in seen:continue
  seen.add(key);pool.append(record(doc,i))
 pool.sort(key=lambda r:hashlib.sha256(('ouro-weight-repair-20261103|'+r['id']).encode()).hexdigest())
 occupied={r['id'] for r in old+selection};unused=[r for r in pool if r['id'] not in occupied]
 extra=render(unused[:192],demos);expanded=old+extra
 assert len({r['id'] for r in expanded})==384
 dq={fingerprint(r['question']) for r in expanded};sq={fingerprint(r['question']) for r in selection};fq={fingerprint(d['question']) for d in demos}
 assert not dq&sq and not (dq|sq)&fq
 screen=sorted(expanded,key=lambda r:hashlib.sha256(('V2-screen64|2021|'+r['id']).encode()).hexdigest())[:64]
 cohort={name:write(name+'.jsonl',rows) for name,rows in [('discovery_expanded',expanded),('discovery_screen',screen)]}
 save('configs/DATA_LOCK.json',dict(discovery_n=384,existing_discovery_n=192,new_previously_unused_n=192,selection_n=192,screen_n=64,demonstrations=25,eligible_unique_train=len(pool),unused_pool_before_expansion=len(unused),cohorts=cohort,selection_sha256=hashlib.sha256((ROOT/'data/selection.jsonl').read_bytes()).hexdigest(),demonstrations_sha256=hashlib.sha256((ROOT/'data/fixed_demonstrations.json').read_bytes()).hexdigest(),discovery_selection_demos_disjoint=True,validation_or_test_opened=False))
 print(json.dumps(dict(status='passed',discovery=384,selection=192,screen=64,extra=192)))
if __name__=='__main__':main()
