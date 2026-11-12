import json,hashlib,pathlib,gzip,os,datetime
P=pathlib.Path('/home/luansiyu/loop-latent');run=P/'orchestration_runs/round2-full-20261111';out=pathlib.Path(__file__).parent;am=P/'research_plans/ROUND_2_TRAIN_DISCLOSURE_AMENDMENT.json';cov=run/'coverage_train.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode();a=json.loads(am.read_text()); assert a['round2_train_coverage_sha256']==sha(cov);c=json.loads(cov.read_text());assert c['status']=='COMPLETE_CUDA_REPLAY_ZERO_TOLERANCE' and c['expected_questions']==c['complete_questions']==1119 and c['replayed_condition_logits']==12309
rows=c['questions'];assert len(rows)==len({q['id'] for q in rows})==1119;assert hashlib.sha256(json.dumps([q['id'] for q in rows],separators=(',',':')).encode()).hexdigest()==a['train_id_order_sha256']
counts={'all':0,'non_demo':0,'demo':0}; names=['train_all.jsonl.gz','train_non_demo.jsonl.gz']; streams=[]
try:
 for n in names:
  raw=(out/n).open('xb');streams.append((raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)))
 for q in rows:
  assert q['status']=='complete';key=hashlib.sha256(q['id'].encode()).hexdigest();p=run/'train'/(key+'.json');assert sha(p)==q['metadata_sha256'];j=json.loads(p.read_text());payload=j['payload'];assert j['payload_sha256']==hashlib.sha256(canonical(payload)).hexdigest();idn=payload['identity'];r=payload['result'];assert idn['id']==q['id'] and idn['split']=='train' and payload['logits_sha256']==q['logits_sha256'];assert {k:idn[k] for k in c['source']}==c['source']; assert set(r['conditions'])=={'K1C_K1R','K1C_K2R','K2C_K1R','K2C_K2R'} and set(r['norm_conditions'])=={'C_K1DIR_K2NORM','C_K2DIR_K1NORM','R_K1DIR_K2NORM','R_K2DIR_K1NORM'}
  row={'id':q['id'],'is_demonstration':idn['is_demonstration'],'metadata_sha256':q['metadata_sha256'],'logits_sha256':q['logits_sha256'],'source_identity':idn,'result':r};b=canonical(row)+b'\n';streams[0][1].write(b);counts['all']+=1
  if idn['is_demonstration']:counts['demo']+=1
  else:streams[1][1].write(b);counts['non_demo']+=1
finally:
 for raw,g in streams:g.close();raw.close()
assert counts=={'all':1119,'non_demo':1094,'demo':25};assert sha(cov)==a['round2_train_coverage_sha256'];info={'status':'TRAIN_ONLY_DISCLOSED_COVERAGE_BOUND','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts':counts,'amendment_sha256':sha(am),'coverage_sha256':sha(cov),'source':c['source'],'artifacts':{n:sha(out/n) for n in names},'note':'validation and test intervention outcomes not read; logits hash anchored to prior coverage, not re-read during export'}
with (out/'export_receipt.json').open('x') as f:json.dump(info,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(info,sort_keys=True))
