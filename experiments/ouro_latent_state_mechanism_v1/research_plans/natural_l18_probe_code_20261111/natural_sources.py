"""Read and bind natural L18 donors to completed CPU/GPU evidence; draft."""
import json,hashlib
from pathlib import Path
import torch
from safetensors.torch import load_file
from full_input import coverage,load as load_natural
ROOT=Path('/home/luansiyu/loop-latent');PLAN=ROOT/'research_plans';RUN=PLAN/'natural_branch_full_run_20261111';COMBINED=PLAN/'NATURAL_BRANCH_INDEPENDENT_COVERAGE_20261111.json';GPU=PLAN/'natural_branch_gpu_replay_20261111.json'
COMBINED_SHA='99a067df530938136618e80893ae7687a9c5668229c453e237197343aaf944b6'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def require(x,s):
 if not x:raise ValueError(s)
def load(split,ident):
 require(split in ('train','validation','test') and isinstance(ident,str) and bool(ident),'identity')
 require(sha(COMBINED)==COMBINED_SHA,'combined coverage changed');c=json.loads(COMBINED.read_text());require(c['status']=='INDEPENDENT_CPU_STATE_AND_GPU_LOGIT_REPLAY_PASSED' and c['counts']=={'train':1119,'validation':299,'test':1172},'combined status')
 cp=PLAN/('natural_branch_'+split+'_exploration_20261111')/(split+'_coverage_preliminary.json');require(sha(cp)==c['splits'][split]['cpu_coverage_sha256'] and sha(GPU)==c['gpu_replay_sha256'],'coverage links')
 cpu=json.loads(cp.read_text());gpu=json.loads(GPU.read_text());cq=[q for q in cpu['questions'] if q['id']==ident];gq=[q for q in gpu['questions'] if q['id']==ident and q['split']==split];require(len(cq)==len(gq)==1,'question absent/duplicate')
 for k in ('natural_metadata_sha256','natural_tensor_sha256','branch_metadata_sha256','branch_tensor_sha256'):require(cq[0][k]==gq[0][k],'coverage question mismatch')
 name=split+'_'+hashlib.sha256(ident.encode()).hexdigest();mp=RUN/(name+'.json');tp=RUN/(name+'.safetensors');require(sha(mp)==cq[0]['branch_metadata_sha256'] and sha(tp)==cq[0]['branch_tensor_sha256'],'source file hash')
 meta=json.loads(mp.read_text());require(meta['id']==ident and meta['split']==split and meta['probe_tensor_sha256']==cq[0]['branch_tensor_sha256'],'source identity')
 nq=[q for q in coverage(split) if q['id']==ident];require(len(nq)==1,'natural question')
 natural=load_natural(split,nq[0]);require(natural['metadata_sha256']==meta['natural_metadata_sha256'] and natural['tensor_sha256']==meta['natural_tensor_sha256'] and natural['R']==meta['R_absolute'] and natural['targets']==meta['targets'] and natural['gold']==meta['gold'] and natural['scores']==meta['prior_scores'],'natural request')
 tensors=load_file(str(tp));states={}
 for u in (0,1):
  vals={k:tensors[f'u{u}_layer17_{k}'].clone() for k in ('x','a')}
  require(all(v.dtype==torch.bfloat16 and v.shape==(1,2048) and torch.isfinite(v).all() for v in vals.values()),'source shape')
  states[u]=vals
 return {'id':ident,'split':split,'R':meta['R_absolute'],'targets':meta['targets'],'gold':meta['gold'],'source_metadata_sha256':sha(mp),'source_tensor_sha256':sha(tp),'natural_metadata_sha256':meta['natural_metadata_sha256'],'natural_tensor_sha256':meta['natural_tensor_sha256'],'prior_scores':meta['prior_scores'],'states':states,'natural':natural,'combined_sha256':COMBINED_SHA}
