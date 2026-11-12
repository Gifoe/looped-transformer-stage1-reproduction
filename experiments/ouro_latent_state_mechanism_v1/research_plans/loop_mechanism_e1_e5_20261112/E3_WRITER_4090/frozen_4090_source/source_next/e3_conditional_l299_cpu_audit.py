"""CPU audit actual branch arithmetic for the fixed three technical cases."""
import json, pathlib, hashlib, torch
from safetensors.torch import load_file
B=pathlib.Path('/root/autodl-tmp/ouro-e1-e3-offload-20261113/results')
P=B/'conditional_l_validation299'; T=B/'reference_validation299'; L=B/'frozen_unrelated_4090.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def th(t): return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def bits(a,b): return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
pending=json.loads((P/'PENDING.json').read_text()); replay=json.loads((P/'CUDA_REPLAY.json').read_text()); assert replay['pending_sha256']==sha(P/'PENDING.json') and replay['status']=='4090_CONDITIONAL_L_VALIDATION299_BARE_PASSED'
ledger=json.loads(L.read_text());mapfile=B/'frozen_unrelated_validation299.jsonl';mapping={x['id']:x for line in mapfile.read_text().splitlines() if (x:=json.loads(line))};result=[]
for i,case in enumerate(pending['cases']):
 stem=hashlib.sha256(case['id'].encode()).hexdigest();mpath=P/(stem+'.json'); tpath=P/(stem+'.safetensors'); assert sha(mpath)==case['metadata_sha256'] and sha(tpath)==case['tensor_sha256']; m=json.loads(mpath.read_text()); t=load_file(str(tpath)); assert sha(L)==m['donor_ledger_sha256']; assert sha(T/(stem+'.safetensors'))==m['reference_tensor_sha256'];ref=load_file(str(T/(stem+'.safetensors'))); checks=0
 for name,e in m['conditions'].items():
  if e['status']!='OK': continue
  branch='a';indices=tuple(range(9,16)); sign=1 if name.startswith('P_') else -1
  for j in indices:
   k=f'{name}_{j:02d}_'; row=e['sites'][str(j)]; before=t[k+'before']; after=t[k+'after']; d=t[k+'reference_delta']; exact=t[k+'exact']; effective=t[k+'effective_delta']; residual=t[k+'cast_residual']; maprow=mapping[case['id']]['sites'][f'{j:02d}_{branch}']
   assert all(th(t[k+x])==row[x+'_sha256'] for x in ('before','after','reference_delta','effective_delta','cast_residual','exact'))
   if 'unrelated' in name: assert th(d)==maprow['unrelated_sha256'] and row['frozen_unrelated_source']==maprow
   else: assert bits(d,ref[f'layer{j:02d}_{branch}_delta']) and th(d)==maprow['reference_sha256']
   proposal=before.double()+sign*d
   if '_norm' in name:
    n=proposal.norm(); assert n!=0 or before.double().norm()==0
    expected=proposal if n==0 else proposal*(before.double().norm()/n)
   else: expected=proposal
   assert torch.allclose(exact,expected,rtol=1e-13,atol=1e-13) and bits(after,exact.to(torch.bfloat16)) and bits(effective,after.double()-before.double()) and bits(residual,after.double()-exact)
   checks+=1
 result.append({'id':case['id'],'site_checks':checks})
out=P/'INDEPENDENT_CPU_MATH_AUDIT.json'; assert not out.exists(); out.write_text(json.dumps({'status':'4090_CONDITIONAL_L_VALIDATION299_CPU_MATH_PASSED','pending_sha256':sha(P/'PENDING.json'),'cuda_replay_sha256':sha(P/'CUDA_REPLAY.json'),'ledger_sha256':sha(L),'script_sha256':sha(pathlib.Path(__file__)),'cases':result,'scope':'CPU recomputation of saved per-site fixed delta, current-state norm control and BF16 casts; no independent model forward.'},indent=2)+'\n'); print(sha(out))
