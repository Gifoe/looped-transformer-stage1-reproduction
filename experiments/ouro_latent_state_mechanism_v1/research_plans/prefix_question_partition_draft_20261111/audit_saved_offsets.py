import json,hashlib,pathlib,datetime
from partition import partition
P=pathlib.Path('/home/luansiyu/loop-latent');O=pathlib.Path(__file__).parent;base=P/'orchestration_runs/run-20261107T053919Z-6d01138e';ledger=P/'research_plans/prompt_boundary_audit_20261111/boundary_ledger.jsonl';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();sourcehash=sha(ledger); rows=[json.loads(x) for x in ledger.open()];stats={}
for split in ('train','validation','test'):
 cp=(P/'orchestration_runs/resume-test-20261111' if split=='test' else base)/('coverage_'+split+'.json');cov=json.loads(cp.read_text());ch=sha(cp);q={x['id']:x for x in cov['questions']};v=[r for r in rows if r['split']==split];assert len(q)==len(v) and set(q)=={r['id'] for r in v};n=0
 for row in v:
  old=q[row['id']];assert old['status']=='complete';f=base/'shards'/split/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json');p=partition(f.read_bytes(),old['metadata_sha256'],row);assert len(p['R'])==1;n+=1
 assert sha(cp)==ch;stats[split]={'n':n,'natural_coverage_sha256':ch}
assert sha(ledger)==sourcehash
r={'status':'ALL_SAVED_OFFSET_PARTITIONS_PASS_INPUTS_ONLY','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'partition_sha256':sha(O/'partition.py'),'audit_sha256':sha(O/'audit_saved_offsets.py'),'boundary_ledger_sha256':sourcehash,'stats':stats,'note':'original natural metadata only; no ROUND_2 intervention values read'}
with (O/'saved_offset_coverage.json').open('x') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(r));print('RECEIPT_SHA256',sha(O/'saved_offset_coverage.json'))
