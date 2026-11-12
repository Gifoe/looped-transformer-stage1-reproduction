"""GPU validation subset. Fail closed on any nonzero discrepancy; no tuning tolerances."""
import argparse,json,sys,os,tempfile
from pathlib import Path
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file,load_file
from transformers import AutoTokenizer,AutoModelForCausalLM
from prompt_spans import make_question,token_spans
from native_capture import capture,candidate_logprob
from manifest_guard import validate_manifest
from cohort_runner import write_new_json
from cohort_runner import bound_hashes,sha,write_json,split_data_path,execute_request
P=Path(__file__).resolve();R=P.parents[2];ROOT=R.parent
sys.path.insert(0,str(R/'experiments/ouro_weight_space_repair_v2/scripts'))
from r10_core import joint_prefix,question_prompt,joint_requests
from runtime import encode_pair
from ouro_cached_scoring import OuroCachedScorer
from r7_inference import target_lp

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=3);ap.add_argument('--split',required=True,choices=['train','validation','test']);ap.add_argument('--output-dir',default=str(P.parent/'results'));a=ap.parse_args()
 output_dir=Path(a.output_dir).resolve();output_dir.relative_to(ROOT.resolve());output_dir.mkdir(parents=True,exist_ok=True)
 for output in (output_dir/f'GPU_PARITY_{a.split}.json',output_dir/f'GPU_GATE_{a.split}.json'):
  if output.exists():raise RuntimeError('refusing to overwrite gate output: '+str(output))
 for name in {a.split,'validation'}:
  validate_manifest(name,split_data_path(name),P.parent/'results/INPUT_MANIFEST_SUMMARY.json',P.parent/'results'/f'INPUT_MANIFEST_{name}.jsonl')
 if not 3<=a.count<=299:raise ValueError('count must be between 3 and 299')
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
 free,total=torch.cuda.mem_get_info()
 if free<40*1024**3 or free/total<0.90:raise RuntimeError('GPU not sufficiently free for non-interfering validation')
 torch.backends.cuda.matmul.allow_tf32=False;torch.set_grad_enabled(False)
 tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True)
 model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
 model.config.use_cache=False;model.model.config.use_cache=False;scorer=OuroCachedScorer(model,4)
 prefix=joint_prefix(json.loads((R/'experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json').read_text()))
 rows=pq.read_table(split_data_path('validation')).to_pylist();records=[]
 ordered=sorted(rows,key=lambda r:(len(tok.encode(prefix+question_prompt(r['question'],r['choices']['text']),add_special_tokens=False)),len(r['choices']['text']),r['id']))
 groups={n:[row for row in ordered if len(row['choices']['text'])==n] for n in (3,4,5)}
 if any(not groups[n] for n in (3,4,5)):raise RuntimeError('validation choice-count coverage unavailable')
 selected=[groups[3][0],groups[4][len(groups[4])//2],groups[5][-1]]
 selected_ids={r['id'] for r in selected}
 for row in ordered:
  if len(selected)>=a.count:break
  if row['id'] not in selected_ids:selected.append(row);selected_ids.add(row['id'])
 rows=selected
 with torch.inference_mode():
  for row in rows:
   prompt,spans=make_question(prefix,row['question'],row['choices']['text'],question_prompt)
   historical_requests=joint_requests({'question':row['question'],'choices':row['choices']['text']},prefix)
   historical_scores=[[] for _ in range(4)];new_scores=[[] for _ in range(4)];common_prefix=None
   for j in range(len(row['choices']['text'])):
    enc=token_spans(tok,prompt,' '+chr(65+j),spans);ids=enc['input_ids'];target=enc['target_ids'];positions=enc['predecessor_positions'];selected=sorted(set(sum(enc['spans'].values(),[])+positions));read_indices=[selected.index(z) for z in positions]
    reference_context,reference_continuation=historical_requests[j]
    if (prompt,' '+chr(65+j))!=(reference_context,reference_continuation):raise RuntimeError('historical request mismatch')
    hp,ht=encode_pair(tok,reference_context,reference_continuation)
    if hp!=enc['context_ids'] or ht!=target or hp+ht[:-1]!=ids or list(range(len(hp)-1,len(hp)+len(ht)-1))!=positions:raise RuntimeError('historical token boundary mismatch')
    if len(target)!=1:raise RuntimeError('gate currently supports single-token targets only')
    if common_prefix is None:common_prefix=ids
    elif common_prefix!=ids:raise RuntimeError('candidate common model input mismatch')
    x=torch.tensor([ids],device='cuda');base=model(x,labels=None,use_cache=False,exit_at_step=3).logits.detach().cpu()
    repeated=model(x,labels=None,use_cache=False,exit_at_step=3).logits.detach().cpu()
    if not torch.equal(base,repeated):raise RuntimeError('repeatability failure; no tolerance has been frozen')
    _,historical,_=scorer._run(ids,positions=positions,use_cache=False)
    torch.cuda.synchronize();production_baseline_bytes=int(torch.cuda.memory_allocated());torch.cuda.reset_peak_memory_stats()
    execution=execute_request(model,enc)
    production_peak_bytes=int(torch.cuda.max_memory_allocated())
    with capture(model,selected,4) as state: read=model(x,labels=None,use_cache=False,exit_at_step=3).logits.detach().cpu()
    if execution['positions']!=selected or execution['events']!=state['events']:raise RuntimeError('production capture event mismatch')
    for layer_name,layer_value in state['layer_states'].items():
     if not torch.equal(execution['tensors'][layer_name],layer_value):raise RuntimeError('production physical layer state mismatch')
    with capture(model,selected,4,identity=True) as identity: after=model(x,labels=None,use_cache=False,exit_at_step=3).logits.detach().cpu()
    if not torch.equal(base,read) or not torch.equal(base,after):raise RuntimeError('hook parity failed')
    # Change only a future token; compare logits at the same predecessor.
    altered=(target[0]+1)%model.config.vocab_size
    if altered==target[0]:raise RuntimeError('invalid causal perturbation')
    future_a=torch.tensor([ids+[target[0]]],device='cuda');future_b=torch.tensor([ids+[altered]],device='cuda')
    readout=torch.tensor([len(ids)-1],device='cuda')
    causal_a=model(future_a,labels=None,use_cache=False,exit_at_step=3,logits_to_keep=readout).logits
    causal_b=model(future_b,labels=None,use_cache=False,exit_at_step=3,logits_to_keep=readout).logits
    if not torch.equal(causal_a,causal_b):raise RuntimeError('causal suffix invariance failed')
    entry={'id':row['id'],'evaluation_split':'validation','target_split':a.split,'candidate':j,'capture_positions':selected,'validation_process_baseline_allocated_bytes':production_baseline_bytes,'validation_process_peak_allocated_bytes':production_peak_bytes,'target_ids':target,'causal_suffix_max_error':float((causal_a.float()-causal_b.float()).abs().max()),'repeated_max_error':float((base.float()-repeated.float()).abs().max()),'hook_max_error':float((base.float()-read.float()).abs().max()),'identity_max_error':float((base.float()-after.float()).abs().max()),'loops':[]}
    for k in range(4):
     old=float(target_lp(model.lm_head(historical[k])[0].float(),target).sum());new=candidate_logprob(model,state['loop_post_norm'][k],target,read_indices)
     if old!=new or execution['scores'][k]!=new:raise RuntimeError('historical or production scorer mismatch')
     if not torch.equal(execution['tensors'][f'loop{k+1}_post_norm'],state['loop_post_norm'][k]):raise RuntimeError('production states mismatch')
     model.model.total_ut_steps=k+1
     try:independent=model(x,labels=None,use_cache=False,exit_at_step=k).logits
     finally:model.model.total_ut_steps=4
     projected=model.lm_head(state['loop_post_norm'][k][read_indices].unsqueeze(0).cuda())
     if not torch.equal(independent[:,positions,:],projected):raise RuntimeError('independent depth mismatch')
     historical_scores[k].append(old);new_scores[k].append(new)
     entry['loops'].append({'k':k+1,'historical':old,'capture':new,'absolute_error':abs(old-new),'depth_max_error':float((independent[:,positions,:].float()-projected.float()).abs().max())})
    with tempfile.TemporaryDirectory(dir=ROOT) as td:
     path=Path(td)/'states.safetensors';original=execution['tensors'];save_file(original,str(path));loaded=load_file(str(path))
     if any(not torch.equal(original[k],loaded[k]) for k in original):raise RuntimeError('storage roundtrip mismatch')
    records.append(entry)
    del independent,projected,causal_a,causal_b,future_a,future_b,x,base,repeated,read,after,historical,state,identity,execution,loaded,original
   gold=row['choices']['label'].index(row['answerKey'])
   for k in range(4):
    oldrank=sorted(range(len(historical_scores[k])),key=lambda i:-historical_scores[k][i]);newrank=sorted(range(len(new_scores[k])),key=lambda i:-new_scores[k][i])
    if oldrank!=newrank:raise RuntimeError('historical ranking mismatch')
    for rec in records[-len(row['choices']['text']):]:rec.setdefault('decision_audit',[]).append({'k':k+1,'gold':gold,'historical_rank':oldrank,'new_rank':newrank,'tie':len(set(historical_scores[k]))<len(historical_scores[k])})
 evidence={'scope':f'real validation subset; batch size one','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'records':records,'max_allowed_error':0.0}
 ep=output_dir/f'GPU_PARITY_{a.split}.json';write_new_json(ep,evidence)
 checks={k:'passed' for k in ('historical_scorer','independent_depths','identity_hook','bf16_roundtrip','repeatability','single_sequence','causal_mask')}
 gate={'status':'passed','checks':checks,'sha256':bound_hashes(a.split),'producer_sha256':sha(P),'evidence_path':str(ep.relative_to(ROOT)),'evidence_sha256':sha(ep),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'scope':'batch=1; fixed sequence; validation subset'}
 write_new_json(output_dir/f'GPU_GATE_{a.split}.json',gate)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  import traceback,datetime
  failure_dir=Path(sys.argv[sys.argv.index('--output-dir')+1]).resolve() if '--output-dir' in sys.argv else P.parent/'results'
  failure_dir.relative_to(ROOT.resolve());failure_dir.mkdir(parents=True,exist_ok=True)
  write_json(failure_dir/('GPU_GATE_FAILURE_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),{'error':repr(exc),'traceback':traceback.format_exc(),'arguments':sys.argv[1:]})
  raise
