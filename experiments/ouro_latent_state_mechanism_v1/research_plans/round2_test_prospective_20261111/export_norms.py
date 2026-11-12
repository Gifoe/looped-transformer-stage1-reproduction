import json,hashlib,pathlib,gzip,math,datetime
P=pathlib.Path('/home/luansiyu/loop-latent');run=P/'orchestration_runs/round2-full-20261111';out=pathlib.Path(__file__).parent;cov=run/'coverage_test.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();c=json.loads(cov.read_text());ch=sha(cov);assert ch=='1cdee583defb6fd580001b532e9fefd11de8f46b7943707ebdb9a30c3d4da4a7' and c['status']=='COMPLETE_CUDA_REPLAY_ZERO_TOLERANCE' and c['complete_questions']==1172
n=0;counts={'K1DIR_K2NORM':0,'K2DIR_K1NORM':0};maxerr={k:0. for k in counts};dest=out/'test_full_norms.jsonl.gz'
with gzip.open(dest,'xb',compresslevel=3) as f:
 for q in c['questions']:
  name=hashlib.sha256(q['id'].encode()).hexdigest();meta=run/'test'/(name+'.json');assert sha(meta)==q['metadata_sha256'];j=json.loads(meta.read_text());pl=j['payload'];r=pl['result'];assert pl['identity']['id']==q['id'] and pl['logits_sha256']==q['logits_sha256'];z=r['norms'];assert set(z)==set(counts)
  for k in z:
   v=z[k];assert set(v)=={'source_norm','reference_norm','actual_bf16_norm','max_abs_actual_minus_reference'} and math.isfinite(v['max_abs_actual_minus_reference']);L=r['input_length']
   for kind in ('source_norm','reference_norm','actual_bf16_norm'):assert len(v[kind])==1 and len(v[kind][0])==L and all(math.isfinite(x) and x>=0 for x in v[kind][0])
   counts[k]+=L;maxerr[k]=max(maxerr[k],v['max_abs_actual_minus_reference'])
  f.write((json.dumps({'id':q['id'],'metadata_sha256':q['metadata_sha256'],'input_length':r['input_length'],'norms':z},sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode());n+=1
assert n==1172 and sha(cov)==ch
receipt={'status':'TEST_FULL_NORM_ARRAYS_COVERAGE_BOUND','questions':n,'vector_counts':counts,'max_abs_recorded_bf16_norm_error':maxerr,'coverage_sha256':ch,'full_norms_sha256':sha(dest),'export_code_sha256':sha(Path(__file__)),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (out/'test_full_norms_receipt.json').open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(receipt));print('RECEIPT_SHA',sha(out/'test_full_norms_receipt.json'),'BYTES',dest.stat().st_size)
