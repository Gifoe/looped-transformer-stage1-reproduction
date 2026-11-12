"""Read historical metadata and verified branch R anchors, without deleted natural tensors."""
import json,hashlib,zipfile,pathlib,torch
from safetensors import safe_open
import full_input,natural_sources,qkv_runner
from source_input import ROOT,ORIG,sha
ZIP=ROOT/'publication/cleanup_receipts/metadata_preservation_20261112.zip';ZIP_SHA='2b19267f48fa42f9cfdc7447cd505f021eb85fd00ef102bd636664b96b24548e'
LCOV=qkv_runner.LCOV;LCOV_SHA=qkv_runner.LCOV_SHA
QGPU=ROOT/'research_plans/NATURAL_QKV_INDEPENDENT_CUDA_REPLAY_20261112.json';QGPU_SHA='e85fdd3071ef85eec3cafa857b09be844ec3dad04189dd21b8940a41c3a75722'
OLD=ROOT/'research_plans/natural_qkv_full_20261112';BRANCH=ROOT/'research_plans/natural_branch_full_run_20261111'
if not __debug__:raise RuntimeError('optimized python forbidden')
def need(x,msg):
 if not x:raise ValueError(msg)
def initialize():
 need(sha(ZIP)==ZIP_SHA and sha(LCOV)==LCOV_SHA and sha(QGPU)==QGPU_SHA,'coverage/zip hash')
 l=json.loads(LCOV.read_text());g=json.loads(QGPU.read_text());need(l['count']==2590 and g['count']==2590 and l['status']=='INDEPENDENT_L18_CUDA_AND_SOURCE_COVERAGE_PASSED' and g['status']=='GPU_QKV_LOGIT_REPLAY_PASSED_NOT_FULL_STATE_COVERAGE','coverage status')
 global L,Q,ZIP_STAT
 L={(x['split'],x['id']):x for x in l['questions']};Q={(x['split'],x['id']):x for x in g['questions']};need(len(L)==len(Q)==2590 and set(L)==set(Q),'coverage IDs');ZIP_STAT=(ZIP.stat().st_size,ZIP.stat().st_mtime_ns)
def load_natural(split,q):
 ident=q['id'];key=hashlib.sha256(ident.encode()).hexdigest();rel='orchestration_runs/run-20261107T053919Z-6d01138e/shards/'+split+'/'+key+'.json';mp=ROOT/rel
 if mp.exists():raw=mp.read_bytes();where='original_metadata'
 else:
  need((ZIP.stat().st_size,ZIP.stat().st_mtime_ns)==ZIP_STAT,'zip stat drift')
  with zipfile.ZipFile(ZIP) as z:raw=z.read(rel)
  where='metadata_preservation_zip'
 need(hashlib.sha256(raw).hexdigest()==q['metadata_sha256'],'metadata SHA');m=json.loads(raw);records=sorted(m['records'],key=lambda x:x['candidate']);ids=records[0]['encoding']['input_ids'];positions=records[0]['captured_positions'];R=records[0]['encoding']['predecessor_positions'];ih=hashlib.sha256(json.dumps(ids).encode()).hexdigest();need(m['id']==ident and m['split']==split and len(R)==1 and R[0] in positions and [x['candidate'] for x in records]==list(range(len(records))) and all(x['input_ids_sha256']==ih and x['encoding']['input_ids']==ids and x['captured_positions']==positions and x['encoding']['predecessor_positions']==R and len(x['encoding']['target_ids'])==1 for x in records),'metadata request')
 link=L[(split,ident)];bk=split+'_'+key;bp=BRANCH/(bk+'.json');bt=bp.with_suffix('.safetensors');need(sha(bp)==link['branch_metadata_sha256'] and sha(bt)==link['branch_tensor_sha256'] and link['natural_metadata_sha256']==q['metadata_sha256'] and link['natural_tensor_sha256']==q['tensor_sha256'],'branch coverage')
 bm=json.loads(bp.read_text());targets=[x['encoding']['target_ids'][0] for x in records];scores=[[x['scores'][u] for x in records] for u in (0,1)];need(bm['id']==ident and bm['split']==split and bm['R_absolute']==R[0] and bm['gold']==m['gold_index'] and bm['targets']==targets and bm['prior_scores']==scores and bm['natural_metadata_sha256']==q['metadata_sha256'] and bm['natural_tensor_sha256']==q['tensor_sha256'],'branch identity')
 states={};norms=[];logits=[]
 with safe_open(str(bt),framework='pt',device='cpu') as f:
  for u in (0,1):
   for i in range(24):
    v=f.get_tensor(f'u{u}_layer{i:02d}_z');need(v.dtype==torch.bfloat16 and tuple(v.shape)==(1,2048),'branch state');states[(u,i)]=v.clone()
   norm=f.get_tensor(f'loop{u+1}_post_norm');logit=f.get_tensor(f'loop{u+1}_logits');need(norm.dtype==torch.bfloat16 and tuple(norm.shape)==(1,2048) and logit.dtype==torch.bfloat16 and tuple(logit.shape)==(1,49152),'branch norm/logit');norms.append(norm.clone());logits.append(logit.clone())
 return {'id':ident,'split':split,'ids':ids,'R':R[0],'gold':m['gold_index'],'targets':targets,'scores':scores,'states':states,'norms':norms,'logits':logits,'metadata_sha256':q['metadata_sha256'],'tensor_sha256':q['tensor_sha256'],'runtime':m['runtime'],'natural_tensor_scope':'historical_not_reread','metadata_source':where,'actual_branch_metadata_sha256':sha(bp),'actual_branch_tensor_sha256':sha(bt)}
def load(split,ident):
 natural_sources.load_natural=load_natural;src=natural_sources.load(split,ident);key=split+'_'+hashlib.sha256(ident.encode()).hexdigest();q=Q[(split,ident)];mp=OLD/(key+'.json');tp=OLD/(key+'.safetensors');need(sha(mp)==q['metadata_sha256'] and sha(tp)==q['tensor_sha256'],'QKV old coverage');m=json.loads(mp.read_text());need(m['split']==split and m['id']==ident and m['natural_metadata_sha256']==src['natural']['metadata_sha256'] and m['input_ids_sha256']==hashlib.sha256(json.dumps(src['natural']['ids']).encode()).hexdigest(),'QKV old identity');return src,m,tp
