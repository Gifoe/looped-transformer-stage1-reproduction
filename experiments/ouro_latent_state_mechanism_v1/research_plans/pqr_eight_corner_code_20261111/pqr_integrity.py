"""DRAFT per-question resume validation; independent final coverage remains separate."""
import json,hashlib,torch
from safetensors.torch import load_file
from pqr_core import NEW,OLD,CORNERS
if not __debug__:raise RuntimeError('optimized Python forbidden')

def validate(path,canonical,sha,expected,old_result,gold,targets,regions,device="cuda"):
 j=json.loads(path.read_text());p=j['payload'];assert j['payload_sha256']==hashlib.sha256(canonical(p)).hexdigest()
 for k,v in expected.items():assert p[k]==v
 tf=path.with_suffix('.safetensors');assert p['logits_sha256']==sha(tf);result=p['result'];assert result['regions']==regions and result['gold']==gold and result['target_ids']==targets and result['fixed_wrong']==old_result['k1_highest_wrong'] and result['anchors']=={'original_four':'exact','native_K1_K2':'exact','last_layer_C':'exact'} and set(result['corners'])==set(CORNERS)
 for c,k in OLD.items():assert result['corners'][c]==old_result['conditions'][k]
 data=load_file(str(tf));assert set(data)=={'corner_'+c for c in NEW}
 for c in NEW:
  z=data['corner_'+c];assert z.dtype==torch.bfloat16 and tuple(z.shape)==(1,49152) and torch.isfinite(z).all();z=z.to(device);scores=[float(z.float().log_softmax(-1).gather(-1,torch.tensor([[t]],device=device)).sum()) for t in targets];p0=max(range(len(scores)),key=lambda i:(scores[i],-i));r=result['corners'][c]
  assert r=={'scores':scores,'prediction':p0,'top_tie':sum(v==scores[p0] for v in scores)>1,'gold_vs_fixed_wrong_gap':scores[gold]-scores[result['fixed_wrong']]}
 return j

def binding_guard(path,expected_sha,sha):
 if sha(path)!=expected_sha:raise ValueError('binding file drift')

def validate_attempt(payload,binding_sha,root,longest,probe,sha):
 from pathlib import Path
 attempt=payload.get('attempt_id');assert isinstance(attempt,str) and len(attempt)==32 and all(x in '0123456789abcdef' for x in attempt)
 path=payload.get('probe_recheck_path');expected=payload.get('probe_recheck_sha256')
 if probe:assert path is None and expected is None;return
 assert isinstance(path,str) and isinstance(expected,str);p=Path(path).resolve();p.relative_to(root.resolve());assert p.name.endswith('_probe_recheck_'+attempt+'.json') and sha(p)==expected
 r=json.loads(p.read_text());assert r['status']=='PQR_LONGEST_PROBE_CUDA_REPLAY_ZERO' and r['binding_sha256']==binding_sha and r['split']==longest['split'] and r['expected_questions']==r['complete_questions']==1 and r['replayed_condition_logits']==4 and len(r['questions'])==1 and r['questions'][0]['id']==longest['id']

def allocate_attempt(parent,name):
 import uuid
 attempt=uuid.uuid4().hex
 return attempt,parent/(name+'_probe_recheck_'+attempt+'.json')

def require_fresh_probe(is_probe,existing_keys):
 if is_probe and existing_keys:raise ValueError('probe resource measurement requires fresh forward; use new attempt directory')
