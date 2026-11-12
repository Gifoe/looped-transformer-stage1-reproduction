import torch,tempfile,pathlib,json,hashlib
from safetensors.torch import save_file
from pqr_probe_guard import probe_pair,exact_probe_logits
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();longest={'id':'probe','split':'test'};bh='a'*64
with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as tmp:
 root=pathlib.Path(tmp);pairs=[]
 for i in range(2):
  d=root/str(i);d.mkdir();key=hashlib.sha256(b'probe').hexdigest();m=d/(key+'.json');m.write_text('{}');t=d/(key+'.safetensors');save_file({'corner_'+c:torch.zeros((1,49152),dtype=torch.bfloat16) for c in ('121','122','211','212')},str(t));r=d/'_receipt.json';r.write_text(json.dumps({'status':'PQR_PROBE_PENDING_REPLAY','split':'test','questions':1,'binding_sha256':bh,'attempt':i,'resources':{'newly_computed_questions':1,'measurement_scope':'model load through fresh full-question forward and saved-output validation; RSS whole process'}}));c=d/'coverage.json';c.write_text(json.dumps({'status':'PQR_LONGEST_PROBE_CUDA_REPLAY_ZERO','binding_sha256':bh,'receipt_sha256':sha(r),'split':'test','expected_questions':1,'complete_questions':1,'replayed_condition_logits':4,'questions':[{'id':'probe','metadata_sha256':sha(m),'logits_sha256':sha(t)}]}));pairs.append((r,c,t))
 for r,c,t in pairs:probe_pair(r,c,bh,longest,sha)
 assert exact_probe_logits(pairs[0][2],pairs[1][2])
 try:probe_pair(pairs[0][0],pairs[1][1],bh,longest,sha)
 except AssertionError:pass
 else:raise AssertionError('mixed probe receipt/coverage accepted')
 changed=root/'changed.safetensors';d={'corner_'+c:torch.zeros((1,49152),dtype=torch.bfloat16) for c in ('121','122','211','212')};d['corner_121'][0,10]=1;save_file(d,str(changed))
 try:exact_probe_logits(pairs[0][2],changed)
 except ValueError:pass
 else:raise AssertionError('changed fresh logits accepted')
print('PQR_PROBE_PAIR_AND_FRESH_LOGITS_CPU_PASS')
