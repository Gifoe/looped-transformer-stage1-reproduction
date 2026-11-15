"""Freeze the ordered numerical sensitivity sample before any C1 intervention."""
import datetime,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
root=Path('/root/autodl-tmp/ouro-k1-k2-mechanism')
out=root/'C1_FP32_NUMERIC_SELECTION.json'
assert not out.exists() and not list((root/'references').glob('*/*.json'))
inputs=[json.loads(line) for line in (root/'inputs/train.jsonl').read_text().splitlines()]
labelpath=Path('/root/autodl-tmp/ouro-e6a/next_step_train_diagnosis/TRAIN_ONLY_INPUTS.jsonl')
gold={q['id']:q['gold'] for q in (json.loads(line) for line in labelpath.read_text().splitlines())}
assert len(inputs)==len(gold)==1094
groups={'rescue':[],'other':[]};evidence={}
for q in inputs:
 ident=q['id'];variant=ident+'__p0_s0'
 path=Path('/root/autodl-tmp/ouro-e7/reencoding/full_capture')/('train_'+hashlib.sha256(variant.encode()).hexdigest()+'.json')
 u=read(path);assert u['p']==u['s']==0 and u['id']==ident and u['targets']==q['targets']
 assert u['input_ids_sha256']==q['input_ids_sha256'] and u['identity']['gpu']=='NVIDIA GeForce RTX 5090'
 preds=[]
 for row in u['native'][:2]:
  v=row['candidate_logits'];preds.append(v.index(max(v)))
 group='rescue' if preds[0]!=gold[ident] and preds[1]==gold[ident] else 'other'
 groups[group].append(ident)
 evidence[ident]={'native_e7_unit_sha256':sha(path),'input_ids_sha256':q['input_ids_sha256'],
    'native_K1_Rlogits_sha256':u['native'][0]['full_logits_sha256'],
    'native_K2_Rlogits_sha256':u['native'][1]['full_logits_sha256'],'native_group':group}
chosen={g:sorted(ids,key=lambda x:hashlib.sha256(('C1-fp32-v1|'+x).encode()).hexdigest())[:4] for g,ids in groups.items()}
selected=chosen['rescue']+chosen['other']
result={'status':'FROZEN_BEFORE_ANY_C1_GPU_OUTCOME','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'rule':'First4 native rescue + first4 other Train by SHA256(C1-fp32-v1|ID); no C1 corrupted outcome read.',
 'native_source':'Completed same-hardware E7 identity captures, exact original inputs and pinned weights/source. Fresh C1 K1/N must bridge these same endpoint hashes before use; never reselect from C1 U.',
 'pool_counts':{g:len(v) for g,v in groups.items()},'groups':chosen,'selected_ids':selected,
 'selected_evidence':{ident:evidence[ident] for ident in selected},
 'source_sha256':sha(__file__),'goldfree_train_inputs_sha256':sha(root/'inputs/train.jsonl'),
 'cohort_freeze_sha256':sha(root/'inputs/COHORT_AND_OPERATOR_FREEZE.json'),
 'isolated_train_labels_sha256':sha(labelpath),'scientific_order_sha256':sha(root/'ASTRA_ORDER_C1_UPSTREAM_TO_L18.md')}
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
