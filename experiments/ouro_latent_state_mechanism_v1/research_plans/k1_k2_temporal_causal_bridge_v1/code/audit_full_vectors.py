"""All retained neural outputs, independently scored on CPU; no inference or fitting."""
import pathlib,json,hashlib,torch,numpy as np,datetime
R=pathlib.Path('/root/ouro-temporal-bridge-sol-20261117');RAW=pathlib.Path('/root/autodl-tmp/ouro-temporal-bridge-sol-20261117/raw');torch.set_num_threads(4);qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};count=0;maxerr=0.
def check(z,r,targets):
 global count,maxerr
 assert z.numel()==49152;v=z.reshape(-1).float().numpy().astype(np.float64);assert hashlib.sha256(z.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==r['full49152_sha256'];assert v[targets].tolist()==r['candidate_logits'];a=v.max();lse=a+np.log(np.exp(v-a).sum());err=max(abs(x-(v[i]-lse)) for x,i in zip(r['candidate_logprobs'],targets));assert err<2e-5;maxerr=max(err,maxerr);ties=np.flatnonzero(v[targets]==v[targets].max()).tolist();assert ties==r['ties'] and r['prediction']==ties[0];count+=1
for p in sorted((R/'results').glob('TB*.json')):
 data=json.loads(p.read_text());raw=torch.load(RAW/(p.stem+'_A.pt'),map_location='cpu',weights_only=False)
 for id,r in data['native_worlds'].items():
  for d,z in raw['native::'+id].items():check(z,r['scores'][str(d)],qs[id]['targets'])
 for u in data['units']:
  for n,heads in raw['unit::'+u['id']].items():
   for d,z in heads.items():check(z,u['conditions'][n]['scores'][str(d)],qs[u['input_W1']]['targets'])
  assert all(u['conditions']['C00']['scores'][d]['full49152_sha256']==u['native_W0']['scores'][d]['full49152_sha256'] for d in ['1','2']);assert all(u['conditions'][n]['cut']['source_exact_checks']==194 for n in ['C00','C01','C10'])
for p in sorted((R/'B_results').glob('*.json')):
 data=json.loads(p.read_text());raw=torch.load(RAW/(p.stem+'_B.pt'),map_location='cpu',weights_only=False)
 for u in data['units']:
  for mask,heads in raw[u['id']].items():
   for d,z in heads.items():check(z,u['masks'][str(mask)]['scores'][str(d)],qs[next(x['input_W1'] for x in json.loads((R/'results'/p.name).read_text())['units'] if x['id']==u['id'])]['targets'])
for p in sorted((R/'C_results').glob('*.json')):
 data=json.loads(p.read_text());raw=torch.load(RAW/(p.stem+'_C.pt'),map_location='cpu',weights_only=False)
 for u in data['units']:
  for name,heads in raw[u['id']].items():
   for d,z in heads.items():check(z,u['conditions'][name]['scores'][str(d)],[330,389,340,422])
  assert u['conditions']['block']['cut']['full_path_block_H_norm_exact'];assert all(u['conditions']['block']['scores'][d]['full49152_sha256']==u['conditions']['Z']['scores'][d]['full49152_sha256'] for d in ['1','2']);assert all(e['G_current_equals_T_asserted'] for n,r in u['conditions'].items() if n not in ['Z','T'] for e in r['cut']['path_edits'])
(R/'FULL_VECTOR_CPU_AUDIT.json').write_text(json.dumps({'status':'PASS','full49152_endpoints':count,'maximum_FP64_logprob_reference_error':maxerr,'scope':'EveryretainedA/B/Cfullvocabularyhash,score,prediction/tie/logprob; exactC00W0andfullpathblockZidentities; source/Gstateassertioncoverage','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));print('FULL_VECTOR_CPU_PASS',count)
