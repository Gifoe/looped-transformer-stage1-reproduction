"""Fail-closed CPU validators for native ABCA draft; not a production driver."""
import hashlib,json,torch

def digest_ids(ids):
 return hashlib.sha256(json.dumps(ids).encode()).hexdigest()

def validate_records(records,selected,prefix):
 if not records or [r['candidate'] for r in sorted(records,key=lambda r:r['candidate'])]!=list(range(len(records))):raise ValueError('candidate order')
 records=sorted(records,key=lambda r:r['candidate']); ids=records[0]['encoding']['input_ids']; pos=records[0]['encoding']['predecessor_positions']
 if len(ids)!=selected['input_length'] or ids[:len(prefix)]!=prefix or digest_ids(ids)!=selected['input_sha256']:raise ValueError('input identity')
 if any(r['encoding']['input_ids']!=ids or r['encoding']['predecessor_positions']!=pos or len(r['encoding']['target_ids'])!=1 for r in records):raise ValueError('request mismatch')
 return ids,pos,[r['encoding']['target_ids'][0] for r in records],[r['scores'][1] for r in records]

def validate_observations(r,prefix_length,length,expected):
 expect=[(u,i) for u in range(2) for i in range(24)]
 if r['layers']!=expect or [x['layer_loop'] for x in r['sdpa']]!=expect or [x['layer_loop'] for x in r['attention']]!=expect:raise ValueError('event sequence')
 if len(r['rope'])!=1 or len(r['states'])!=2:raise ValueError('capture count')
 positions=r['rope'][0]['position_ids']
 if not torch.equal(positions,torch.arange(length).unsqueeze(0)):raise ValueError('position ids')
 rope=r['rope'][0]['cos_sin']
 if not isinstance(rope,tuple) or len(rope)!=2 or r['rope'][0]['source_devices']!=[expected['device']]*2:raise ValueError('rope arity/device')
 if any(tuple(t.shape)!=(1,length,expected['head_dim']) or str(t.dtype)!=expected['dtype'] or not torch.isfinite(t).all() for t in rope):raise ValueError('rope shape/dtype/finite')
 for a,z in zip(r['attention'],r['sdpa']):
  if not a['past_is_none'] or a['use_cache'] is not False:raise ValueError('past')
  if a['attention_mask'] is not None or z['attn_mask'] is not None or z['is_causal'] is not True or z['dropout_p']!=0.0 or not z['completed']:raise ValueError('unverified causal backend')
  ae=a['position_embeddings']
  if not isinstance(ae,tuple) or len(ae)!=2 or any(tuple(t.shape)!=(1,length,expected['head_dim']) or str(t.dtype)!=expected['dtype'] or not torch.isfinite(t).all() for t in ae):raise ValueError('attention rope')
  if len(z['qkv'])!=3:raise ValueError('QKV count')
  for q,heads in zip(z['qkv'],(expected['q_heads'],expected['kv_heads'],expected['kv_heads'])):
   if q!={'shape':[1,heads,length,expected['head_dim']],'dtype':expected['dtype'],'device':expected['device']}:raise ValueError('QKV identity')
  if any(not torch.equal(x,y) for x,y in zip(a['position_embeddings'],r['rope'][0]['cos_sin'])):raise ValueError('rope mismatch')
 for u,v in enumerate(r['states']):
  t=v['prefix']
  if v['loop']!=u or tuple(t.shape)!=(1,prefix_length,expected['hidden_size']) or str(t.dtype)!=expected['dtype'] or v['source_device']!=expected['device'] or not torch.isfinite(t).all():raise ValueError('state')
 return True

def compare(a,b):
 if a.shape!=b.shape or a.dtype!=b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('incompatible tensors')
 af=a.float();bf=b.float();diff=(af-bf).abs();flat=diff.reshape(-1);index=int(flat.argmax())
 return {'bitwise_equal':bool(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))), 'unequal_elements':int(torch.count_nonzero(a!=b)), 'max_absolute':float(flat[index]), 'max_flat_index':index,'max_relative_to_b':float((diff/bf.abs().clamp_min(torch.finfo(torch.float32).tiny)).max())}
