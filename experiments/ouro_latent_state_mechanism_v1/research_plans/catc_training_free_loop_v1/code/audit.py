import pathlib,json,torch,hashlib,datetime
R=pathlib.Path('/root/ouro-catc-sol-20261116');torch.set_num_threads(4);qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};count=0;maxerr=0.;files=[]
for p in sorted((R/'results').glob('*.json')):
 row=json.loads(p.read_text());q=qs[row['id']];t=torch.load(p.with_suffix('.pt'),weights_only=False)
 for rec in row['records']:
  for key,score in rec['scores'].items():
   z=t[rec['name']][int(key)];assert z.shape==(1,49152) and z.dtype==torch.bfloat16;assert hashlib.sha256(z.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==score['logits_sha256'];v=z.flatten()[q['targets']].float().tolist();assert v==score['candidate_logits'];ties=[i for i,x in enumerate(v) if x==max(v)];assert ties==score['ties'] and ties[0]==score['prediction'];lp=z.double().log_softmax(-1).flatten()[q['targets']].tolist();e=max(abs(a-b) for a,b in zip(lp,score['scores']));assert e<1e-5;maxerr=max(maxerr,e);count+=1
  for requested,cap,scale,actual,before in rec['norms']:assert actual<=cap+1e-12 and actual/before<=.1+1e-12
 files.append({'path':str(p.with_suffix('.pt').relative_to(R)),'bytes':p.with_suffix('.pt').stat().st_size,'sha256':hashlib.sha256(p.with_suffix('.pt').read_bytes()).hexdigest()})
assert len(files)==1094
for p in sorted((R/'technical').glob('*.pt')):files.append({'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(R/'RAW_TENSOR_MANIFEST.json').write_text(json.dumps(files,indent=2));(R/'INDEPENDENT_CPU_AUDIT.json').write_text(json.dumps({'status':'PASS','questions':1094,'full49152_endpoints':count,'max_logprob_vs_FP64_error':maxerr,'all_actual_correction_norms_bounded':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));print('CPU_AUDIT_PASS',count)
