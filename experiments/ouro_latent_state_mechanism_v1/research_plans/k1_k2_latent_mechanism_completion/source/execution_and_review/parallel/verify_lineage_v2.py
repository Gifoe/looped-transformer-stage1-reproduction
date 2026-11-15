import hashlib,json,pathlib,time
R=pathlib.Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(pathlib.Path(p).read_text())
old=R/'async_io/EXECUTION_AMENDMENT.json';new=R/'parallel/EXECUTION_AMENDMENT.json'
oldsha,newsha=sha(old),sha(new)
a=read(new);science=R/'B2_FULL_SOURCE_FREEZE.json'
retained={x['id']:x for x in read(R/'parallel/TRANSITION_RECEIPT.json')['retained']}
results=[]
for stage in ('train64','train1030','validation299'):
 d=R/'b2_data/treatments'/stage
 while not (R/'b2_analysis'/stage/'COMPLETE.json').exists():
  assert read(R/'B2_QUEUE_STATE.json')['status']!='FAILED_STOPPED'
  time.sleep(30)
 p=read(d/'PENDING.json');audit=read(d/'INDEPENDENT_COVERAGE_AUDIT.json')
 assert p['status']=='COMPLETE' and audit['status']=='PASS' and audit['pending_sha256']==sha(d/'PENDING.json')
 counts={'original':0,'async':0,'isolated_parallel':0,'retained_unchanged':0}
 for c in p['cases']:
  mp=d/c['metadata_file'];assert sha(mp)==c['metadata_sha256'];m=read(mp)
  if m.get('execution_revision')=='isolated_parallel_v1':
   assert m['parallel_execution_amendment_sha256']==newsha
   sp=R/m['parallel_spec_path'];assert sha(sp)==m['parallel_spec_sha256']
   spec=read(sp);assert spec['mode']=='science' and spec['execution_amendment_sha256']==newsha
   assert m['id'] in spec['assignments'][m['worker_index']]
   counts['isolated_parallel']+=1
  elif 'execution_amendment_sha256' in m:
   assert m['execution_amendment_sha256']==oldsha and m['execution_revision']=='async_cpu_writer_v1'
   counts['async']+=1
  else:counts['original']+=1
  if stage=='train1030' and c['id'] in retained:
   assert c['metadata_sha256']==retained[c['id']]['metadata_sha256'] and c['tensor_raw_sha256']==retained[c['id']]['tensor_raw_sha256']
   counts['retained_unchanged']+=1
  assert m['source_freeze_sha256']==a['science_freeze_sha256'] and m['producer_identity']['source_freeze_sha256']==a['science_freeze_sha256']
 if stage=='train1030':assert counts['retained_unchanged']==len(retained)
 assert sha(science)==a['science_freeze_sha256']
 assert all(sha(R/k)==v for k,v in read(science)['sources'].items())
 assert all(sha(R/k)==v for k,v in a['sources'].items())
 result=dict(status='PASS',stage=stage,counts=counts,pending_sha256=sha(d/'PENDING.json'),audit_sha256=sha(d/'INDEPENDENT_COVERAGE_AUDIT.json'),execution_amendment_sha256=newsha,script_sha256=sha(__file__),scope='Independent source and execution lineage, not a substitute for original tensor audits.')
 (R/'parallel'/(stage+'_LINEAGE.json')).write_text(json.dumps(result,indent=2));results.append(result)
(R/'parallel/ALL_LINEAGE_COMPLETE.json').write_text(json.dumps(dict(status='PASS',stages=results),indent=2))
print('ALL_PARALLEL_LINEAGE_PASS',flush=True)
