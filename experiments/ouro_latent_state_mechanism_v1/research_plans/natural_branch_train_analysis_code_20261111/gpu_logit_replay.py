"""Blind independent replay: no branch effect summaries or lens inspection."""
import os,json,fcntl,hashlib
from pathlib import Path
import torch,technical,full_input,state_capture,full_runner_draft,source_input,decompose
from safetensors.torch import load_file
from safetensors import safe_open
from source_input import ROOT,ORIG,sha,initial_sources
from full_input import coverage
from technical import score
from state_capture import bits
RUN=ROOT/'research_plans/natural_branch_full_run_20261111';DEST=ROOT/'research_plans/natural_branch_gpu_replay_20261111.json'
EXPECTED_CODES={'technical.py':'537231dbd895221d1775daf31ac6a172f19f97a28cbccb2c066b7e99b20c4dcd','full_input.py':'0da5cc5db704290daba15d8db23cdd6ee31c636f8adb6b01a23ef0a748717986','state_capture.py':'f757d947c3eea7966a86f3a17cc5008418488c3e2915ec685cdebb8f30839c1e','source_input.py':'d52ef07a08011cf28cb78c2883cb86b047b319317a163ce61145fb630963a050','decompose.py':'cb730f7b0c09d3e8074a488955ed525d6b50acc29d26b13f23c4405872549243'}
RUNNERS=('3441948e63da0b1ca02dc59636113715915254a944f32cb46fee0c10ce248d1b','d19f7fd0b6f1b6b03a73ce821851c1dc4a475b3ffa0138db27ddd19baa11be72','490ff360f270e67c9516dbeb509c4ed839007261a88485fedc231d5594ddda5b','70669efd01c2cfd981c31879c0b8b2ebcba5bcd7a81f9378273aa46e8315893d')
def require(b,s):
 if not b:raise ValueError(s)
def main():
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  receipt=json.loads((RUN/'receipt.json').read_text());require(receipt['status']=='FULL_OBSERVATION_PENDING_INDEPENDENT_COVERAGE' and receipt['counts']=={'train':1119,'validation':299,'test':1172},'collector incomplete')
  sources=initial_sources();start={k:sha(v) for k,v in sources.items()};modules={'technical':Path(technical.__file__),'full_input':Path(full_input.__file__),'state_capture':Path(state_capture.__file__),'source_input':Path(source_input.__file__),'decompose':Path(decompose.__file__),'reservation_runner':Path(full_runner_draft.__file__)};code_hashes={k:sha(v) for k,v in modules.items()};require(all(code_hashes[k]==EXPECTED_CODES[k+'.py'] for k in ('technical','full_input','state_capture','source_input','decompose')) and code_hashes['reservation_runner']==RUNNERS[-1],'import code'); geo=json.loads((ROOT/'research_plans/natural_r_geometry_20261111/source.json').read_text());coverage_hashes={s:geo[s]['coverage_sha256'] for s in ('train','validation','test')};require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'cuda/bf16')
  free=torch.cuda.mem_get_info()[0]
  if free>40*1024**3+512*1024**2:
   from full_runner_draft import reserve_to_target
   reserve_to_target()
  else:print('GPU_RESERVATION_UNAVAILABLE_CONTINUE_REPLAY',free,flush=True)
  runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()};out=[]
  for split in ('train','validation','test'):
   for q in coverage(split):
    ident=q['id'];base=hashlib.sha256(ident.encode()).hexdigest();natural=ORIG/'shards'/split/(base+'.json');nt=natural.with_suffix('.safetensors');mp=RUN/(split+'_'+base+'.json');tp=mp.with_suffix('.safetensors')
    require(sha(natural)==q['metadata_sha256'] and sha(nt)==q['tensor_sha256'],'natural SHA');m=json.loads(mp.read_text());require(m['id']==ident and m['split']==split and m['probe_tensor_sha256']==sha(tp) and m['natural_metadata_sha256']==q['metadata_sha256'] and m['natural_tensor_sha256']==q['tensor_sha256'],'branch SHA')
    sh=m['source_hashes'];require(sh['code_full_runner_draft.py'] in RUNNERS and all(sh['code_'+k]==v for k,v in EXPECTED_CODES.items()) and all(sh[k]==v for k,v in start.items()),'branch source code')
    orig=json.loads(natural.read_text());records=sorted(orig['records'],key=lambda r:r['candidate']);targets=[r['encoding']['target_ids'][0] for r in records];expected=[[r['scores'][k] for r in records] for k in (0,1)];ids=records[0]['encoding']['input_ids']
    require([r['candidate'] for r in records]==list(range(len(records))) and all(len(r['encoding']['target_ids'])==1 and r['encoding']['input_ids']==ids for r in records) and m['targets']==targets and m['prior_scores']==expected,'request')
    require(all(orig['runtime'][k]==runtime[k] for k in runtime),'runtime')
    def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    pos=torch.arange(len(ids),dtype=torch.long);ph=digest(pos.unsqueeze(0));ch=digest(pos);events=m['attention'];require(len(events)==48 and [x.get('loop_layer') for x in events]==[[u,i] for u in (0,1) for i in range(24)],'attention order');reference=None
    for row in events:
     require(row['position_ids']=={'shape':[1,len(ids)],'dtype':'torch.int64','sha256':ph} and row['cache_position']=={'shape':[len(ids)],'dtype':'torch.int64','sha256':ch} and row['past_is_none'] is True and row['use_cache'] is False,'positions/past')
     current={k:row[k] for k in ('attention_mask','position_ids','cache_position','rope','past_is_none','use_cache')}
     require(isinstance(current['rope'],list) and len(current['rope'])==2 and all(current['rope']),'rope')
     if reference is None:reference=current
     else:require(current==reference,'attention event drift')
    with safe_open(str(tp),framework='pt',device='cpu') as f,safe_open(str(nt),framework='pt',device='cpu') as old:
     for u in (1,2):
      z=f.get_tensor(f'loop{u}_logits');z0=old.get_tensor(f'candidate0_loop{u}_lm_head_logits');require(z.dtype==torch.bfloat16 and z.shape==(1,49152) and torch.isfinite(z).all() and bits(z,z0),'logits bits')
      got=score(z.to('cuda'),targets);require(got==expected[u-1],'GPU score mismatch')
    out.append({'split':split,'id':ident,'natural_metadata_sha256':sha(natural),'natural_tensor_sha256':q['tensor_sha256'],'branch_metadata_sha256':sha(mp),'branch_tensor_sha256':m['probe_tensor_sha256']})
    if len(out)%100==0:print('REPLAYED',len(out),flush=True)
  require(len(out)==2590 and {k:sha(v) for k,v in sources.items()}==start and {k:sha(v) for k,v in modules.items()}==code_hashes,'final source');require(not DEST.exists(),'replay receipt exists')
  result={'status':'GPU_LOGIT_REPLAY_PASSED_NOT_FULL_STATE_COVERAGE','count':len(out),'runtime':runtime,'collector_receipt_sha256':sha(RUN/'receipt.json'),'source_hashes':start,'imported_code_sha256':code_hashes,'natural_coverage_sha256':coverage_hashes,'script_sha256':sha(Path(__file__)),'questions':out}
  with DEST.open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  print('GPU_LOGIT_REPLAY_COMPLETE',len(out),flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
