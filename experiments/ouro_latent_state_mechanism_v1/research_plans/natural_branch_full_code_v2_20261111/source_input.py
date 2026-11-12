"""Load input-only A/B/C natural originals with full source verification."""
import hashlib,json,torch
from pathlib import Path
from safetensors import safe_open
ROOT=Path('/home/luansiyu/loop-latent');ORIG=ROOT/'orchestration_runs/run-20261107T053919Z-6d01138e';RES=ROOT/'orchestration_runs/resume-test-20261111';SEL=ROOT/'research_plans/PREFIX_NATIVE_ABCA_SELECTION.json'
EXPECTED={'selection':'1dcf857e8f9b05297fcb140a629c05f278707095de071747f62d2bc44d14cfac','coverage':'df164143c5921971c54a314ef560b3e3cc8566ff13636a931690efe53943d5e1','model':'58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af','source':'c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413','config':'ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def initial_sources():
 paths={'selection':SEL,'coverage':RES/'coverage_test.json','model':ROOT/'model/model.safetensors','source':ROOT/'model/modeling_ouro.py','config':ROOT/'model/config.json'}
 for k,p in paths.items():
  if sha(p)!=EXPECTED[k]:raise ValueError('source hash '+k)
 return paths

def load_one(label):
 if label not in ('A','B','C'):raise ValueError('selection')
 if sha(SEL)!=EXPECTED['selection'] or sha(RES/'coverage_test.json')!=EXPECTED['coverage']:raise ValueError('selection/coverage drift')
 selected=json.loads(SEL.read_text())['selection'][label]
 if selected['split']!='test':raise ValueError('split')
 coverage=json.loads((RES/'coverage_test.json').read_text());matches=[q for q in coverage['questions'] if q['id']==selected['id']]
 if len(matches)!=1 or matches[0]['status']!='complete' or matches[0]['errors']:raise ValueError('coverage')
 q=matches[0];key=hashlib.sha256(selected['id'].encode()).hexdigest();mp=ORIG/'shards/test'/(key+'.json');tp=mp.with_suffix('.safetensors')
 if sha(mp)!=q['metadata_sha256'] or sha(tp)!=q['tensor_sha256']:raise ValueError('shard hash')
 meta=json.loads(mp.read_text());records=sorted(meta['records'],key=lambda x:x['candidate']);ids=records[0]['encoding']['input_ids'];positions=records[0]['captured_positions'];R=records[0]['encoding']['predecessor_positions'];ih=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
 if ih!=selected['input_sha256'] or len(ids)!=selected['input_length'] or len(R)!=1 or R[0] not in positions or [x['candidate'] for x in records]!=list(range(len(records))):raise ValueError('selection binding')
 if any(x['input_ids_sha256']!=ih or x['encoding']['input_ids']!=ids or x['captured_positions']!=positions or x['encoding']['predecessor_positions']!=R or len(x['encoding']['target_ids'])!=1 for x in records):raise ValueError('candidate binding')
 idx=positions.index(R[0]);states={};norms=[];logits=[]
 with safe_open(str(tp),framework='pt',device='cpu') as f:
  for u in (0,1):
   for i in range(24):
    v=f.get_slice(f'candidate0_loop{u+1}_layer{i:02d}')
    if v.get_dtype()!='BF16' or v.get_shape()!=[len(positions),2048]:raise ValueError('state shape')
    states[(u,i)]=v[idx:idx+1,:].clone()
   v=f.get_slice(f'candidate0_loop{u+1}_post_norm');
   if v.get_dtype()!='BF16' or v.get_shape()!=[len(positions),2048]:raise ValueError('norm shape')
   norms.append(v[idx:idx+1,:].clone());z=f.get_tensor(f'candidate0_loop{u+1}_lm_head_logits')
   if z.dtype!=torch.bfloat16 or tuple(z.shape)!=(1,49152):raise ValueError('logits')
   logits.append(z.clone())
 return {'id':selected['id'],'label':label,'ids':ids,'R':R[0],'saved_row':idx,'gold':meta['gold_index'],'targets':[x['encoding']['target_ids'][0] for x in records],'scores':[[x['scores'][k] for x in records] for k in (0,1)],'states':states,'norms':norms,'logits':logits,'metadata_sha256':sha(mp),'tensor_sha256':sha(tp)}
