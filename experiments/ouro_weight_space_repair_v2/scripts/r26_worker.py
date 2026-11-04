"""Exactly thirty original uncached forwards, no interventions/hooks."""
import time,inspect,importlib,os
from r26_common import ROOT,LOCK,read,sha,digest,jsonlines,save,gate
from r26_protocol import build_inputs,variants

def tensor_digest(t):
 import hashlib
 return hashlib.sha256(t.detach().contiguous().view(__import__('torch').uint8).cpu().numpy().tobytes()).hexdigest()
def compare(a,b):
 import torch
 assert a.shape==b.shape and a.dtype==b.dtype
 d=a.double()-b.double();assert torch.isfinite(d).all()
 unequal=(a.contiguous().view(torch.uint8).reshape(a.numel(),a.element_size())!=b.contiguous().view(torch.uint8).reshape(b.numel(),b.element_size())).any(dim=1)
 return dict(exact=not bool(unequal.any()),shape=list(a.shape),dtype=str(a.dtype),sha256_A=tensor_digest(a),sha256_B=tensor_digest(b),unequal_elements=int(unequal.sum()),max_abs=float(d.abs().max()),RMS=float(d.square().mean().sqrt()))
def main():
 started=time.monotonic();spec=gate()
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Ouro-1.4B'),local_files_only=True,use_fast=True)
 from r26_preflight import audit as input_audit
 audit=input_audit(tok);assert audit==read('results/r26_CPU_INPUT_PREFLIGHT.json')
 plan=audit['plan'];save('results/r26_INPUT_AUDIT.json',audit)
 from runtime import load_model,check_budget
 from ouro_cached_scoring import OuroCachedScorer
 from weight_edits import tensor_sha256
 import torch
 import importlib.metadata
 assert importlib.metadata.version('torch')==spec['runtime_identity']['torch'] and importlib.metadata.version('transformers')==spec['runtime_identity']['transformers']
 torch.set_grad_enabled(False)
 tok,model=load_model();assert sha('models/Ouro-1.4B/model.safetensors')==spec['model_identity']['safetensors_sha256']
 assert model.config.vocab_size==49152 and model.config._attn_implementation=='sdpa'
 module=importlib.import_module(type(model).__module__);assert sha(inspect.getfile(module))==spec['runtime_identity']['model_source_sha256']
 import transformers.integrations.sdpa_attention as sdpa
 import transformers.masking_utils as masks
 assert sha(inspect.getfile(sdpa))==spec['runtime_identity']['sdpa_sha256'];assert sha(inspect.getfile(masks))==spec['runtime_identity']['masking_sha256']
 assert model.config.attention_dropout==0 and model.config.sliding_window is None and all(x=='full_attention' for x in model.config.layer_types)
 assert len(model.model.layers)==24 and all(layer.self_attn.is_causal and layer.self_attn.sliding_window is None for layer in model.model.layers)
 assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
 before={n:tensor_sha256(p) for n,p in model.named_parameters()};records=[];vectors={}
 def retain(t):
  import numpy as np,hashlib
  a=t.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16) if t.dtype==torch.bfloat16 else t.detach().contiguous().cpu().numpy()
  key=hashlib.sha256((str(t.dtype)+str(tuple(t.shape))).encode()+a.tobytes()).hexdigest()
  if key in vectors:assert np.array_equal(vectors[key],a)
  else:vectors[key]=a.copy()
  return key
 torch.cuda.reset_peak_memory_stats()
 for c in plan['contexts']:
  scorer=OuroCachedScorer(model,c['depth']);outputs={};input_records=[]
  for name,tokens,positions in variants(c['tokens'],c['cuts']):
   check_budget(5);assert time.monotonic()<float(os.environ['R26_SOFT_MONOTONIC'])
   states=scorer._run(tokens,positions=positions,use_cache=False)[1];assert len(states)==c['depth'];packed={}
   for k,h in enumerate(states,1):
    assert h.dtype==torch.bfloat16 and h.shape==(1,len(positions),2048)
    for i,pos in enumerate(positions):
     single=h[:,i:i+1,:];assert single.shape[:2]==(1,1)
     z=model.lm_head(single).float();assert z.shape==(1,1,49152) and torch.isfinite(z).all()
     packed[(k,pos)]=(single.detach().clone(),z.detach().clone())
   outputs[name]=packed;input_records.append(dict(variant=name,input_ids=tokens,input_sha256=digest(tokens),shape=[1,len(tokens)],positions=positions))
  comparisons=[]
  for name in ['B1','B2','A_repeat','C1','C2']:
   for (k,pos),(h,z) in outputs[name].items():
    if name in ['B1','B2'] and pos!=c['cuts'][int(name[-1])-1]:continue
    ah,az=outputs['A'][(k,pos)];sm=compare(ah,h);hm=compare(az,z)
    sm.update(vector_A=retain(ah),vector_B=retain(h));hm.update(vector_A=retain(az),vector_B=retain(z))
    comparisons.append(dict(variant=name,depth=k,position=pos,state=sm,head=hm))
  record=dict(context=c['name'],depth=c['depth'],cuts=c['cuts'],inputs=input_records,comparisons=comparisons,complete=True);records.append(record);save('results/r26_PROGRESS.json',dict(completed_contexts=len(records),target_contexts=5,elapsed_seconds=time.monotonic()-started))
  del outputs,scorer
 after={n:tensor_sha256(p) for n,p in model.named_parameters()};assert before==after
 import numpy as np
 with (ROOT/'results/r26_VECTORS.npz').open('xb') as f:np.savez(f,**vectors)
 save('results/r26_RAW.json',dict(records=records))
 save('results/r26_COMPLETE.json',dict(status='complete',contexts=5,forwards=30,protocol_sha256=sha(LOCK),raw_sha256=sha('results/r26_RAW.json'),vectors_sha256=sha('results/r26_VECTORS.npz'),unique_vectors=len(vectors),input_audit_sha256=sha('results/r26_INPUT_AUDIT.json'),all_parameter_hashes_before=before,all_parameter_hashes_after=after,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started,source_bound_call_contract='Unchanged _run(...use_cache=False,capture_raw=False), no hooks/mask/backend patch',heldout_accessed=False))
if __name__=='__main__':main()
