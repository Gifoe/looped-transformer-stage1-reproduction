"""Independent full-vocabulary endpoint and cohort audit using retained tensors only."""
import pathlib,json,hashlib,torch,time
R=pathlib.Path('/root/ouro-computation-sol-20261116');torch.set_num_threads(4)
def th(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
count=0;maxerr=0.;files=[]
def check(z,r,targets):
 global count,maxerr
 assert z.shape==(1,49152) and z.dtype==torch.bfloat16 and torch.isfinite(z).all();assert th(z)==r['logits_sha256']
 values=z.reshape(-1)[targets].float().tolist();assert values==r['candidate_logits'];ties=[i for i,v in enumerate(values) if v==max(values)];assert ties==r['ties'] and ties[0]==r['prediction']
 lp=z.double().log_softmax(-1).reshape(-1)[targets].tolist();err=max(abs(a-b) for a,b in zip(lp,r['scores']));assert err<1e-5;maxerr=max(err,maxerr);count+=1
for required in ['A_validation_COMPLETE.json','B_heldout_COMPLETE.json','B_NATIVE_SERIALIZATION_REPAIR.json']:assert (R/required).exists()
A={q['id']:q for q in map(json.loads,(R/'A_INPUTS.jsonl').read_text().splitlines())};seen=[]
for p in sorted((R/'data/A').glob('*.json')):
 m=json.loads(p.read_text());seen.append(m['id']);q=A[m['id']];t=load(p.with_suffix('.pt'));assert t['B'].shape==(1,len(q['ids']),2048)
 for r in m['records']:
  key={'K1':'K1_logits','native':'native_logits'}.get(r['condition'],r['condition']+'_logits');check(t[key],r,q['targets'])
 files.append({'path':str(p.relative_to(R)),'tensor_bytes':p.with_suffix('.pt').stat().st_size,'tensor_sha256':hashlib.sha256(p.with_suffix('.pt').read_bytes()).hexdigest()})
assert len(seen)==len(set(seen))==1393 and set(seen)==set(A)
B={q['id']:q for q in map(json.loads,(R/'B_INPUTS.jsonl').read_text().splitlines())};bun=0
for p in sorted((R/'data/B').glob('*.json')):
 m=json.loads(p.read_text());old=load(p.with_suffix('.pt'));restored=load(R/'data/B_native_restored'/p.with_suffix('.pt').name)
 def native(id):return restored[id] if id in restored else old['native::'+id] if 'native::'+id in old else old[id]
 for u in m['units']:
  bun+=1;rec={r['condition']:r for r in u['records']};q=B[u['receiver']];rr=native(u['receiver']);dd=native(u['donor']);ss=native(u['same_variable']);ee=native(u['text_edit']);lo,hi=q['source_span'];assert torch.equal(rr['B'][:,lo:hi].view(torch.uint8),ss['B'][:,lo:hi].view(torch.uint8));assert torch.equal(dd['B'][:,lo:hi].view(torch.uint8),ee['B'][:,lo:hi].view(torch.uint8));patch=old.get('patch::'+u['unit_id'],old.get(u['unit_id']))
  for condition,r in rec.items():
   z=rr['K1_logits'] if condition=='K1' else rr['native_logits'] if condition=='native' else ee['K1_logits'] if condition=='text_edit_K1' else ee['native_logits'] if condition=='text_edit_K2' else patch[condition]['logits'];check(z,r,q['targets'])
  assert torch.equal(patch['same_variable']['logits'].view(torch.uint8),rr['native_logits'].view(torch.uint8))
 files.append({'path':str(p.relative_to(R)),'tensor_bytes':p.with_suffix('.pt').stat().st_size,'tensor_sha256':hashlib.sha256(p.with_suffix('.pt').read_bytes()).hexdigest(),'restored_native_sha256':hashlib.sha256((R/'data/B_native_restored'/p.with_suffix('.pt').name).read_bytes()).hexdigest()})
assert bun==640
(R/'RAW_TENSOR_MANIFEST.json').write_text(json.dumps(files,indent=2))
result={'status':'PASS','A_questions':len(seen),'B_units':bun,'audited_full49152_endpoints':count,'candidate_logits_predictions_ties':'exact','maximum_FP64_reference_vs_GPU_FP32_logprob_absolute_error':maxerr,'causal_future_mapping_source_equality':'all640 units exact','same_variable_null_full49152':'all640 units exact','raw_tensor_manifest':'RAW_TENSOR_MANIFEST.json','scope':'No new forward, no fitting; complete frozen populations, native serialization repair independently checked by original fullendpoint hashes.'}
(R/'INDEPENDENT_CPU_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
