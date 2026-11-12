"""Natural full-cohort source reader for branch observation; review-only."""
import json,hashlib,torch
from pathlib import Path
from safetensors import safe_open
from source_input import ROOT,ORIG,RES,sha
SOURCE=ROOT/'research_plans/natural_r_geometry_20261111/source.json'
SOURCE_SHA='8e22e6ef7d5cd920a7f5ac4faae0d7df31f002a094346e289cb325f9d5443ed2'
def coverage(split):
 if sha(SOURCE)!=SOURCE_SHA:raise ValueError('geometry source')
 source=json.loads(SOURCE.read_text());path=(RES if split=='test' else ORIG)/('coverage_'+split+'.json')
 if sha(path)!=source[split]['coverage_sha256']:raise ValueError('coverage hash')
 data=json.loads(path.read_text());n={'train':1119,'validation':299,'test':1172}[split];c=data['counts']
 if c['expected_questions']!=n or c['complete_questions']!=n or c['failed_questions'] or c['missing_questions'] or data['score_replay']['status']!='passed' or data['score_replay']['tolerance']!=0:raise ValueError('coverage status')
 rows=data['questions']
 if len(rows)!=n or len({r['id'] for r in rows})!=n or any(r['status']!='complete' or r['split']!=split or r['errors'] for r in rows):raise ValueError('coverage IDs')
 return rows

def load(split,q):
 ident=q['id'];key=hashlib.sha256(ident.encode()).hexdigest();mp=ORIG/'shards'/split/(key+'.json');tp=mp.with_suffix('.safetensors')
 if sha(mp)!=q['metadata_sha256'] or sha(tp)!=q['tensor_sha256']:raise ValueError('shard hash')
 m=json.loads(mp.read_text());records=sorted(m['records'],key=lambda r:r['candidate']);ids=records[0]['encoding']['input_ids'];positions=records[0]['captured_positions'];R=records[0]['encoding']['predecessor_positions'];ih=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
 if m['id']!=ident or m['split']!=split or len(R)!=1 or R[0] not in positions or [r['candidate'] for r in records]!=list(range(len(records))):raise ValueError('record identity')
 if any(r['input_ids_sha256']!=ih or r['encoding']['input_ids']!=ids or r['captured_positions']!=positions or r['encoding']['predecessor_positions']!=R or len(r['encoding']['target_ids'])!=1 for r in records):raise ValueError('candidate alignment')
 idx=positions.index(R[0]);states={};norms=[];logits=[]
 with safe_open(str(tp),framework='pt',device='cpu') as f:
  for u in (0,1):
   for i in range(24):
    v=f.get_slice(f'candidate0_loop{u+1}_layer{i:02d}')
    if v.get_dtype()!='BF16' or v.get_shape()!=[len(positions),2048]:raise ValueError('state shape')
    states[(u,i)]=v[idx:idx+1,:].clone()
   v=f.get_slice(f'candidate0_loop{u+1}_post_norm')
   if v.get_dtype()!='BF16' or v.get_shape()!=[len(positions),2048]:raise ValueError('norm shape')
   norms.append(v[idx:idx+1,:].clone());z=f.get_tensor(f'candidate0_loop{u+1}_lm_head_logits')
   if z.dtype!=torch.bfloat16 or tuple(z.shape)!=(1,49152):raise ValueError('logits')
   logits.append(z.clone())
 return {'id':ident,'split':split,'ids':ids,'R':R[0],'saved_row':idx,'gold':m['gold_index'],'targets':[r['encoding']['target_ids'][0] for r in records],'scores':[[r['scores'][k] for r in records] for k in (0,1)],'states':states,'norms':norms,'logits':logits,'metadata_sha256':sha(mp),'tensor_sha256':sha(tp),'runtime':m['runtime']}
