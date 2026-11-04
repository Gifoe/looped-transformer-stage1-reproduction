"""One fixed at-most33-forward diagnostic; no ARC score readers."""
import time,os,inspect,importlib
from r27_exec_common import ROOT,LOCK,read,sha,save,gate,Vectors
from r27_exec_protocol import ordered_instances,PATCHES,label_metrics
from r27_exec_hooks import SourceReplay

def main():
 started=time.monotonic();spec=gate()
 from transformers import AutoTokenizer
 from r27_exec_preflight import audit
 tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Ouro-1.4B'),local_files_only=True,use_fast=True);input_audit=audit(tok);assert input_audit==read('results/r27_EXEC_INPUT_PREFLIGHT.json');save('results/r27_EXEC_INPUT_AUDIT.json',input_audit)
 rows=ordered_instances(read('configs/r27_CPU_STRUCTURE_MANIFEST.json'))
 from runtime import load_model,check_budget
 from ouro_cached_scoring import OuroCachedScorer
 from weight_edits import tensor_sha256
 import torch,importlib.metadata
 torch.set_grad_enabled(False);tok,model=load_model();identity=spec['runtime_identity']
 assert sha(inspect.getfile(importlib.import_module(type(model).__module__)))==identity['model_source_sha256']
 import transformers.integrations.sdpa_attention as sdpa,transformers.masking_utils as masks
 assert sha(inspect.getfile(sdpa))==identity['sdpa_sha256'] and sha(inspect.getfile(masks))==identity['masking_sha256']
 assert all(importlib.metadata.version(k)==identity[k] for k in ['torch','transformers'])
 assert model.config._attn_implementation=='sdpa' and model.config.attention_dropout==0 and model.config.sliding_window is None and len(model.model.layers)==24 and all(l.self_attn.is_causal and l.self_attn.sliding_window is None for l in model.model.layers)
 assert all(t=='full_attention' for t in model.config.layer_types) and not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
 expected_width=model.config.num_key_value_heads*model.config.head_dim;assert expected_width==spec['projection_width']
 before={n:tensor_sha256(p) for n,p in model.named_parameters()};assert len(before)==269 and before==read('results/r26_COMPLETE.json')['all_parameter_hashes_after']
 vectors=Vectors();baselines=[];patches=[];calls=[];source_pairs=[];invalid=[];forwards=0;controller=None;fatal=None
 def run(row,scorer,full=False):
  nonlocal forwards
  check_budget(5);assert time.monotonic()<float(os.environ['R27_EXEC_SOFT_MONOTONIC'])
  ids=row['input_ids'];positions=range(len(ids)) if full else [len(ids)-1];states=scorer._run(ids,positions=positions,use_cache=False)[1];forwards+=1;assert len(states)==4
  query=[s[:,-1:,:] for s in states];z=model.lm_head(query[3]).float();assert z.shape==(1,1,49152)
  flags=[torch.isfinite(s).all() for s in states]+[torch.isfinite(z).all()];assert bool(torch.stack(flags).all().item())
  return states,query,z
 def pack(row,query,z):return dict(table=row['table'],query_row=row['query_row'],version=row['version'],input_ids_sha256=row['input_ids_sha256'],input_shape=row['input_shape'],source_positions=row['source_positions'],expected_label=row['expected_label'],label_ids={x['label']:x['token_id'] for x in row['sources']},query_states=[vectors.keep(h) for h in query],fullV=vectors.keep(z))
 try:
  plain=OuroCachedScorer(model,4);states,q,z=run(rows[0],plain);noop=pack(rows[0],q,z);calls.append(dict(number=forwards,phase='no_hook',table=rows[0]['table'],query_row=rows[0]['query_row'],version='A'));del states,q,z,plain
  captures={};Bfull_refs={}
  with SourceReplay(model) as controller:
   scorer=OuroCachedScorer(model,4);controller.bind(scorer)
   for row in rows:
    controller.configure('capture',row['source_positions'],len(row['input_ids']));states,q,z=run(row,scorer,full=row['version']=='B');clock=controller.verify();record=pack(row,q,z);record['clock']=clock
    captures[(row['table'],row['query_row'],row['version'])]={k:dict(v) for k,v in controller.captured.items()}
    record['donors']={kind:[vectors.keep(captures[(row['table'],row['query_row'],row['version'])][kind][i]) for i in range(24)] for kind in ['K','V']}
    assert all(v['shape']==[1,4,expected_width] for vv in record['donors'].values() for v in vv)
    if row['version']=='B':
     refs=[vectors.keep(s) for s in states[:3]];record['first3_full_states']=refs;Bfull_refs[(row['table'],row['query_row'])]=refs
    baselines.append(record);calls.append(dict(number=forwards,phase='A',table=row['table'],query_row=row['query_row'],version=row['version'],clock=clock));save('results/r27_EXEC_PROGRESS.json',dict(forwards=forwards,phase='A',baselines=len(baselines),patches=0,elapsed_seconds=time.monotonic()-started));del states,q,z
   no_op_exact=noop['query_states']==baselines[0]['query_states'] and noop['fullV']==baselines[0]['fullV']
   if not no_op_exact:invalid.append('No-hook/capture query states or fullV not bit exact')
   for table in [1,2]:
    query_rows=[r['query_row'] for r in baselines if r['table']==table and r['version']=='A']
    for version in ['A','B']:
     a,b=[r for r in baselines if r['table']==table and r['version']==version]
     for kind in ['K','V']:
      for layer in range(24):
       ar=a['donors'][kind][layer];br=b['donors'][kind][layer];exact=ar==br;source_pairs.append(dict(table=table,version=version,kind=kind,layer=layer,A=ar,B=br,exact=exact))
       if not exact:invalid.append('Source-query independence not exact')
   assert len(source_pairs)==192
   phaseA_pass=False
   if not invalid:
    for r in baselines:r['metrics']=label_metrics(vectors.arrays[r['fullV']['key']],r['label_ids'],r['expected_label'])
    phaseA_pass=all(r['metrics']['expected_strictly_highest'] for r in baselines)
   if phaseA_pass:
    for row in [r for r in rows if r['version']=='B']:
     baseline=next(r for r in baselines if r['table']==row['table'] and r['query_row']==row['query_row'] and r['version']=='B');first_query=next(r['query_row'] for r in rows if r['table']==row['table'])
     for name,kind,version,shuffle in PATCHES:
      donor=captures[(row['table'],first_query,version)][kind];permutation=(1,2,3,0) if shuffle else (0,1,2,3)
      controller.configure('patch',row['source_positions'],len(row['input_ids']),kind,donor,permutation);states,q,z=run(row,scorer,full=True);clock=controller.verify();record=pack(row,q,z);record.update(patch=name,clock=clock,donor_vectors=[vectors.keep(donor[i]) for i in range(24)],donor_table=row['table'],donor_query_row=first_query,donor_version=version,donor_kind=kind)
      refs=[vectors.keep(s) for s in states[:3]];first3_exact=refs==Bfull_refs[(row['table'],row['query_row'])];record.update(first3_full_states=refs,baseline_first3_full_states=Bfull_refs[(row['table'],row['query_row'])],first3_full_exact=first3_exact)
      if not first3_exact:invalid.append('First3 full-sequence states changed')
      if name in ['V_identity','K_identity']:
       identity_exact=record['query_states']==baseline['query_states'] and record['fullV']==baseline['fullV'];record['identity_exact']=identity_exact
       if not identity_exact:invalid.append('Identity replay query states/fullV changed')
      patches.append(record);calls.append(dict(number=forwards,phase='B',table=row['table'],query_row=row['query_row'],patch=name,clock=clock));save('results/r27_EXEC_PROGRESS.json',dict(forwards=forwards,phase='B',baselines=8,patches=len(patches),elapsed_seconds=time.monotonic()-started));del states,q,z
 except BaseException as error:fatal=repr(error);invalid.append(fatal)
 finally:
  cleanup=getattr(controller,'cleanup',dict(handles_removed=True,original_hooks_restored=True,errors=[]))
  all_hooks_empty=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules());cleanup['all_model_hooks_empty']=all_hooks_empty
  if not all_hooks_empty:invalid.append('Hook cleanup incomplete')
  after={n:tensor_sha256(p) for n,p in model.named_parameters()}
  if before!=after:invalid.append('Parameter hash changed')
  if not invalid and len(patches)==24:
   for r in patches:r['metrics']=label_metrics(vectors.arrays[r['fullV']['key']],r['label_ids'],r['expected_label'])
  vectors.write();raw=dict(no_hook=noop if 'noop' in locals() else None,baselines=baselines,patches=patches,source_pairs=source_pairs,forward_records=calls,cleanup=cleanup,invalid_reasons=invalid,no_op_exact=locals().get('no_op_exact',False),phaseA_pass=locals().get('phaseA_pass',False));save('results/r27_EXEC_RAW.json',raw)
  status='invalid' if invalid else ('complete_phaseB' if len(patches)==24 else 'complete_phaseA_not_admitted')
  save('results/r27_EXEC_COMPLETE.json',dict(status=status,forwards=forwards,baselines=len(baselines),patches=len(patches),protocol_sha256=sha(LOCK),raw_sha256=sha('results/r27_EXEC_RAW.json'),vectors_sha256=sha('results/r27_EXEC_VECTORS.npz'),input_audit_sha256=sha('results/r27_EXEC_INPUT_AUDIT.json'),all_parameter_hashes_before=before,all_parameter_hashes_after=after,cleanup=cleanup,invalid_reasons=invalid,elapsed_seconds=time.monotonic()-started,source_bound_runtime_identity=identity,call_scope='Original uncached _run, native4, no explicit mask/KV; internal SDPA not intercepted',peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
 if fatal:raise RuntimeError(fatal)
if __name__=='__main__':main()
