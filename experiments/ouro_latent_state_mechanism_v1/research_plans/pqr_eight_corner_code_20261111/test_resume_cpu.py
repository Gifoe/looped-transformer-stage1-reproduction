import torch,json,hashlib,tempfile,pathlib,copy
from safetensors.torch import save_file
from pqr_integrity import validate
from pqr_core import NEW,OLD
canon=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();targets=[1,2];gold=0;wrong=1;regions={'P':[0],'Q':[1],'R':[2]};z=torch.randn(1,49152,dtype=torch.bfloat16);scores=[float(z.float().log_softmax(-1).gather(-1,torch.tensor([[t]])).sum()) for t in targets];pred=max(range(2),key=lambda i:(scores[i],-i));entry={'scores':scores,'prediction':pred,'top_tie':scores[0]==scores[1],'gold_vs_fixed_wrong_gap':scores[0]-scores[1]};old={'k1_highest_wrong':wrong,'conditions':{v:copy.deepcopy(entry) for v in OLD.values()}};result={'regions':regions,'gold':gold,'target_ids':targets,'fixed_wrong':wrong,'anchors':{'original_four':'exact','native_K1_K2':'exact','last_layer_C':'exact'},'corners':{k:copy.deepcopy(entry) for k in ('111','112','121','122','211','212','221','222')}};expected={'id':'a','binding_sha256':'a'*64};
with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as d:
 p=pathlib.Path(d)/'a.json';tf=p.with_suffix('.safetensors');save_file({'corner_'+c:z.clone() for c in NEW},str(tf));payload={**expected,'result':result,'logits_sha256':sha(tf)}
 def write(x):p.write_bytes(canon({'payload':x,'payload_sha256':hashlib.sha256(canon(x)).hexdigest()}))
 def check(x):return validate(p,canon,sha,x,old,gold,targets,regions,device='cpu')
 def reject(f):
  try:f()
  except (AssertionError,ValueError,KeyError):return
  raise AssertionError('corruption accepted')
 write(payload);check(expected);reject(lambda:check({'id':'wrong','binding_sha256':'a'*64}))
 bad=copy.deepcopy(payload);bad['result']['corners']['121']['scores'][0]+=1;assert all(bad['result']['corners'][c]==payload['result']['corners'][c] for c in ('111','112','122','211','212','221','222'));write(bad);reject(lambda:check(expected))
 bad=copy.deepcopy(payload);bad['result']['regions']={'P':[0,1],'Q':[],'R':[2]};write(bad);reject(lambda:check(expected))
 write(payload);raw=bytearray(tf.read_bytes());raw[-1]^=1;tf.write_bytes(raw);reject(lambda:check(expected))
print('PQR_RESUME_CPU_CORRUPTION_PASS (CPU diagnostic, not CUDA gate)')
