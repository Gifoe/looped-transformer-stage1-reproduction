"""Independent saved evidence replay; CUDA required for production use."""
import json,torch
from pathlib import Path
from safetensors.torch import load_file
from integrity import validate_observations
from persistence import sha,tensor_hash

def replay(out,name,entry,manifest_sha,expected,device):
 out=Path(out);m=json.loads((out/(name+'.json')).read_text());tf=out/(name+'.safetensors')
 if m['name']!=name or m['manifest_sha256']!=manifest_sha or m['tensor_sha256']!=sha(tf):raise ValueError('receipt provenance')
 identity={k:v for k,v in entry.items() if k not in ('ids','scores')}
 if m['identity']!=identity or m['scores']!=entry['scores']:raise ValueError('natural identity')
 t=load_file(str(tf))
 if set(t)!={'loop0','loop1','rope_cos','rope_sin','positions','logits'}:raise ValueError('tensor keys')
 rope=(t['rope_cos'],t['rope_sin']);hashes=[tensor_hash(x) for x in rope];att=[]
 for a in m['attention']:
  if a['rope_hashes']!=hashes:raise ValueError('attention rope binding')
  att.append({'layer_loop':tuple(a['layer_loop']),'attention_mask':a['attention_mask'],'past_is_none':a['past_is_none'],'use_cache':a['use_cache'],'position_embeddings':rope})
 r={'layers':[tuple(x) for x in m['layers']],'sdpa':[dict(x,layer_loop=tuple(x['layer_loop'])) for x in m['sdpa']],'attention':att,'rope':[{'position_ids':t['positions'],'cos_sin':rope,'source_devices':m['rope_devices']}],'states':[{'loop':u,'prefix':t['loop'+str(u)],'source_device':m['state_devices'][u]} for u in range(2)]}
 validate_observations(r,expected['prefix_length'],len(entry['ids']),expected)
 z=t['logits']
 if z.dtype!=torch.bfloat16 or tuple(z.shape)!=(1,expected['vocab_size']) or not torch.isfinite(z).all():raise ValueError('logits')
 z=z.to(device);scores=[float(z.float().log_softmax(-1).gather(-1,torch.tensor([[target]],device=device)).sum()) for target in entry['targets']]
 if scores!=entry['scores']:raise ValueError('score replay')
 return {'tensor_sha256':sha(tf),'receipt_sha256':sha(out/(name+'.json')),'scores_zero_difference':True,'replay_device':str(device)}
