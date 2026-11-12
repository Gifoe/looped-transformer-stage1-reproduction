"""Round 2 anchor draft: real GPU parity assertions; no full intervention output."""
import argparse,json,hashlib,sys,time,os,subprocess
if not __debug__:raise RuntimeError("optimized Python is forbidden for GPU anchors")
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from round2_full_sequence import full_boundary
from cohort_runner import sha,bound_hashes
from ledger_guard import require_production_ledger
P=Path(__file__).resolve();ROOT=P.parents[2].parent
ORIG=ROOT/'orchestration_runs/run-20261107T053919Z-6d01138e';RESUME=ROOT/'orchestration_runs/resume-test-20261111'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out).resolve();out.relative_to(ROOT.resolve());assert not out.exists()
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
 free,total=torch.cuda.mem_get_info();assert free>20*1024**3
 span=json.loads((ORIG/'span_review.json').read_text());ids=[('validation',x['id']) for x in span['records']];probe=json.loads((ORIG/'GPU_RESOURCE_PROBE.json').read_text());case=probe['cases']['longest_input'];ids.append((case['split'],case['id']));assert len(ids)==4 and len({id for _,id in ids})==4
 bindings={};code_hashes={'anchor':sha(P),'hook':sha(P.with_name('round2_full_sequence.py'))};interpreter=Path(sys.executable).resolve();interpreter_sha=sha(interpreter);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parents[2],text=True).strip()
 frozen=sorted((P.parent/'environment_pip_freeze.txt').read_text().splitlines())
 def environment_guard():
  current=sorted(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,timeout=60).splitlines());assert current==frozen;assert sha(interpreter)==interpreter_sha;assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parents[2],text=True).strip()==commit
 environment_guard()
 for split in {split for split,_ in ids}:
  root=RESUME if split=='test' else ORIG;cp=root/f'coverage_{split}.json';cov=json.loads(cp.read_text());require_production_ledger(cov);gate=json.loads((ORIG/'gates'/f'GPU_GATE_{split}.json').read_text());actual=bound_hashes(split);assert actual==gate['sha256'];bindings[split]={'coverage_path':str(cp),'coverage_sha256':sha(cp),'bound_hashes':actual}
 def check_binding(split):
  environment_guard()
  assert sha(P)==code_hashes['anchor'] and sha(P.with_name('round2_full_sequence.py'))==code_hashes['hook'];assert bound_hashes(split)==bindings[split]['bound_hashes'];assert sha(Path(bindings[split]['coverage_path']))==bindings[split]['coverage_sha256']
 torch.cuda.reset_peak_memory_stats()
 model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False
 if model.training or model.config._attn_implementation!='sdpa' or next(model.parameters()).dtype!=torch.bfloat16:raise RuntimeError('incorrect execution settings')
 results=[]
 with torch.inference_mode():
  for split,ident in ids:
   check_binding(split)
   coverage=json.loads(((RESUME if split=='test' else ORIG)/f'coverage_{split}.json').read_text());ledger=next(q for q in coverage['questions'] if q['id']==ident);assert ledger['status']=='complete'
   name=hashlib.sha256(ident.encode()).hexdigest();path=ORIG/'shards'/split/(name+'.json');tensor=path.with_suffix('.safetensors');assert sha(path)==ledger['metadata_sha256'] and sha(tensor)==ledger['tensor_sha256']
   meta=json.loads(path.read_text());assert meta['id']==ident and meta['sha256']==bound_hashes(split);records=sorted(meta['records'],key=lambda x:x['candidate']);assert len(records)>=3 and [r['candidate'] for r in records]==list(range(len(records)))
   ids0=records[0]['encoding']['input_ids'];pos=records[0]['encoding']['predecessor_positions'];assert pos and all(r['encoding']['input_ids']==ids0 and r['encoding']['predecessor_positions']==pos and len(r['encoding']['target_ids'])==1 for r in records)
   actual_input_sha=hashlib.sha256(json.dumps(ids0).encode()).hexdigest();assert all(r['input_ids_sha256']==actual_input_sha for r in records)
   x=torch.tensor([ids0],dtype=torch.long,device='cuda');targets=[r['encoding']['target_ids'][0] for r in records]
   def forward(depth,**hook_options):
    norms=[]
    def on_norm(module,args,out):
     if not torch.is_tensor(out) or out.ndim!=3:raise RuntimeError('invalid norm output')
     norms.append(out[0,pos,:].detach().clone())
    handle=model.model.norm.register_forward_hook(on_norm)
    try:
     with full_boundary(model,depth,**hook_options) as states:y=model(x,labels=None,use_cache=False,exit_at_step=depth-1)
    finally:handle.remove()
    if len(norms)!=depth:raise RuntimeError('wrong norm invocation count')
    projected=model.lm_head(norms[-1].unsqueeze(0))[0]
    native=y.logits[0,pos,:].detach().clone();del y
    if projected.dtype!=torch.bfloat16 or native.dtype!=torch.bfloat16 or len(pos)!=1:raise RuntimeError('invalid scorer shape/dtype')
    if not torch.isfinite(projected).all() or not torch.isfinite(native).all():raise RuntimeError('nonfinite logits')
    if not torch.equal(projected,native):
     err=float((projected.float()-native.float()).abs().max().item())
     raise RuntimeError('full-head vs production-projected head mismatch; max_abs='+str(err))
    values=[]
    for target in targets:
     token=torch.tensor([target],device=projected.device,dtype=torch.long)
     values.append(float(projected.float().log_softmax(-1).gather(-1,token[:,None]).sum()))
    return states,values,projected.detach().clone()
   st,s1,z1=forward(1);k1=st[0];assert s1==[r['scores'][0] for r in records],(split,ident,'K1 baseline',s1,[r['scores'][0] for r in records])
   st,s2,z2=forward(2);k2=st[1];assert s2==[r['scores'][1] for r in records],(split,ident,'K2 baseline',s2,[r['scores'][1] for r in records])
   _,si,zi=forward(2,condition='K2C_K2R',k1=k1,k2_reference=k2,readout=pos);assert si==s2 and torch.equal(zi,z2),'K2 identity mismatch'
   _,sa,za=forward(2,condition='K1C_K1R',k1=k1,k2_reference=k2,readout=pos);assert sa==s1 and torch.equal(za,z1),'full K1 replacement mismatch'
   t1,s1last,z1last=forward(1,layer_index=23);t2,s2last,z2last=forward(2,layer_index=23);assert s1last==s1 and s2last==s2 and torch.equal(z1last,z1) and torch.equal(z2last,z2)
   _,sn,zn=forward(2,condition='K1C_K2R',k1=t1[0],k2_reference=t2[1],readout=pos,layer_index=23);assert sn==s2 and torch.equal(zn,z2),'last layer C-only negative control mismatch'
   check_binding(split)
   results.append({'split':split,'id':ident,'input_sha256':actual_input_sha,'metadata_path':str(path),'metadata_sha256':sha(path),'tensor_path':str(tensor),'tensor_sha256':sha(tensor),'coverage_path':bindings[split]['coverage_path'],'coverage_sha256':bindings[split]['coverage_sha256'],'input_length':len(ids0),'readout_positions':pos,'candidate_count':len(records),'baseline_K1_scores':s1,'baseline_K2_scores':s2,'production_projected_head_vs_full_head':'exact','K2_identity':'exact','K1_all_anchor':'exact','last_layer_C_only':'exact'});print('ANCHOR_PASS',split,ident,flush=True)
 report={'status':'ANCHORS_PASSED_ONLY','note':'not full intervention and not causal evidence','cases':results,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'source_sha256':code_hashes,'bindings':bindings,'cuda_version':torch.version.cuda,'free_before_bytes':free,'total_bytes':total,'source_commit':commit,'interpreter_path':str(interpreter),'interpreter_sha256':interpreter_sha,'pip_freeze_path':str(P.parent/'environment_pip_freeze.txt'),'pip_freeze_sha256':sha(P.parent/'environment_pip_freeze.txt'),'allow_tf32':torch.backends.cuda.matmul.allow_tf32}
 with out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
