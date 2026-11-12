import tempfile,json
from pathlib import Path
from provenance import sha
from finalize import publish
with tempfile.TemporaryDirectory() as td:
 d=Path(td);ledger={}
 for name in ('A1','B','C','A2'):
  (d/(name+'.json')).write_text(name);(d/(name+'.safetensors')).write_text(name)
  ledger[name]={'scores_zero_difference':True,'replay_device':'cuda','receipt_sha256':sha(d/(name+'.json')),'tensor_sha256':sha(d/(name+'.safetensors'))}
 (d/'replay_ledger.json').write_text(json.dumps(ledger));h=sha(d/'replay_ledger.json');calls=[]
 def guard():
  calls.append(1)
  if len(calls)==2:raise ValueError('source drift during publish')
 try:publish(d,{}, {},'a'*64,h,ledger,guard)
 except ValueError:pass
 else:raise AssertionError('guard missed')
 assert not (d/'comparison.json').exists() and len(calls)==2
 publish(d,{}, {},'a'*64,h,ledger,lambda:None);assert (d/'comparison.json').exists()
print('FINAL_GUARD_TOY_PASS; NOT GPU')
