"""Probe receipt/coverage/shards pairing and cross-attempt exact logits."""
import json,hashlib,torch
from pathlib import Path
from safetensors.torch import load_file
if not __debug__:raise RuntimeError('optimized Python forbidden')
def probe_pair(receipt_path,coverage_path,binding_sha,longest,sha):
 receipt_path=Path(receipt_path);coverage_path=Path(coverage_path);r=json.loads(receipt_path.read_text());c=json.loads(coverage_path.read_text());assert r['status']=='PQR_PROBE_PENDING_REPLAY' and r['split']==longest['split'] and r['questions']==1 and r['binding_sha256']==binding_sha
 rs=r['resources'];assert rs['newly_computed_questions']==1 and rs['measurement_scope']=='model load through fresh full-question forward and saved-output validation; RSS whole process'
 assert c['status']=='PQR_LONGEST_PROBE_CUDA_REPLAY_ZERO' and c['binding_sha256']==binding_sha and c['receipt_sha256']==sha(receipt_path) and c['split']==longest['split'] and c['expected_questions']==c['complete_questions']==1 and c['replayed_condition_logits']==4 and len(c['questions'])==1 and c['questions'][0]['id']==longest['id']
 q=c['questions'][0];name=hashlib.sha256(longest['id'].encode()).hexdigest();meta=receipt_path.parent/(name+'.json');tensor=receipt_path.parent/(name+'.safetensors');assert sha(meta)==q['metadata_sha256'] and sha(tensor)==q['logits_sha256'];return r,c,tensor

def exact_probe_logits(a,b):
 x=load_file(str(a));y=load_file(str(b));expected={'corner_121','corner_122','corner_211','corner_212'}
 if set(x)!=expected or set(y)!=expected:raise ValueError('probe keys')
 if any(x[k].dtype!=torch.bfloat16 or tuple(x[k].shape)!=(1,49152) or x[k].dtype!=y[k].dtype or x[k].shape!=y[k].shape or not torch.equal(x[k],y[k]) for k in expected):raise ValueError('reviewed/fresh probe logits drift')
 return True
