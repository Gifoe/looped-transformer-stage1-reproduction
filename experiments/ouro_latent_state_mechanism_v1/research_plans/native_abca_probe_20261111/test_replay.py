import tempfile,json
from pathlib import Path
from test_capture_core import obs,scores,z,entry,cfg
from persistence import save_capture
from replay import replay
cfg=dict(cfg,prefix_length=3,vocab_size=16)
entry=dict(entry,id='sample',metadata_sha256='m',tensor_sha256='t')
identity={k:v for k,v in entry.items() if k not in ('ids','scores')}
with tempfile.TemporaryDirectory() as d:
 save_capture(d,'A1',identity,obs,scores,z,'a'*64)
 assert replay(d,'A1',entry,'a'*64,cfg,'cpu')['scores_zero_difference']
 from safetensors.torch import load_file,save_file
 import hashlib
 tf=Path(d)/'A1.safetensors';p=Path(d)/'A1.json';original=tf.read_bytes();meta=json.loads(p.read_text())
 for mode in ('short_prefix','extra_key','missing_key','wrong_vocab'):
  tf.write_bytes(original);x=load_file(str(tf))
  if mode=='short_prefix':x['loop0']=x['loop0'][:,:2,:].contiguous();x['loop1']=x['loop1'][:,:2,:].contiguous()
  elif mode=='extra_key':x['extra']=x['loop0'].clone()
  elif mode=='missing_key':del x['loop1']
  else:x['logits']=x['logits'][:,:15].contiguous()
  save_file(x,str(tf));v=dict(meta,tensor_sha256=hashlib.sha256(tf.read_bytes()).hexdigest());p.write_text(json.dumps(v))
  try:replay(d,'A1',entry,'a'*64,cfg,'cpu')
  except ValueError:pass
  else:raise AssertionError(mode)
 tf.write_bytes(original);p.write_text(json.dumps(meta));m=json.loads(p.read_text());m['attention'][0]['past_is_none']=False;p.write_text(json.dumps(m))
 try:replay(d,'A1',entry,'a'*64,cfg,'cpu')
 except ValueError:pass
 else:raise AssertionError('past altered')
print('REPLAY_TOY_PASS CPU only')
