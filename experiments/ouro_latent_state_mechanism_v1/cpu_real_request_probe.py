"""Bounded CPU-only real 25-shot technical probe; never a GPU gate or cohort."""
import sys,time,json,tempfile
from pathlib import Path
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file,load_file
from transformers import AutoTokenizer,AutoModelForCausalLM
from cohort_runner import PROJECT,REPO,split_data_path,execute_request,write_json,sha
from prompt_spans import make_question,token_spans
sys.path.insert(0,str(REPO/'experiments/ouro_weight_space_repair_v2/scripts'))
from r10_core import joint_prefix,question_prompt,joint_requests
from runtime import encode_pair
from ouro_cached_scoring import OuroCachedScorer
from r7_inference import target_lp
P=Path(__file__).resolve();torch.set_num_threads(2);torch.set_grad_enabled(False)
row=pq.read_table(split_data_path('validation')).to_pylist()[0];demo=json.loads((REPO/'experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json').read_text());prefix=joint_prefix(demo);prompt,spans=make_question(prefix,row['question'],row['choices']['text'],question_prompt);tok=AutoTokenizer.from_pretrained(PROJECT/'model',local_files_only=True,trust_remote_code=True)
enc=token_spans(tok,prompt,' A',spans);reference=joint_requests({'question':row['question'],'choices':row['choices']['text']},prefix)[0];hp,ht=encode_pair(tok,*reference)
if reference!=(prompt,' A') or hp!=enc['context_ids'] or ht!=enc['target_ids']:raise RuntimeError('historical input mismatch')
m=AutoModelForCausalLM.from_pretrained(PROJECT/'model',trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval();m.config.use_cache=False;m.model.config.use_cache=False
if m.config._attn_implementation!='sdpa' or any(p.dtype!=torch.bfloat16 or p.device.type!='cpu' for p in m.parameters() if p.is_floating_point()) or torch.get_num_threads()!=2:raise RuntimeError('BF16/SDPA/CPU/thread invariant failed')
start=time.monotonic();m.model.total_ut_steps=1
try:
 short=m(torch.tensor([[330,389]]),labels=None,use_cache=False,exit_at_step=0,logits_to_keep=1).logits
 if short.dtype!=torch.bfloat16 or not bool(torch.isfinite(short).all()):raise RuntimeError('short BF16 SDPA failed')
 tick=time.monotonic();small=m(torch.tensor([enc['input_ids'][:128]]),labels=None,use_cache=False,exit_at_step=0,logits_to_keep=1).logits;small_time=time.monotonic()-tick
finally:m.model.total_ut_steps=4
L=len(enc['input_ids']);d=m.config.hidden_size;cost=lambda n:n*12*d*d+2*n*n*d;estimate=small_time*8*cost(L)/cost(128);print(json.dumps({'stage':'preflight','id':row['id'],'tokens':L,'threads':torch.get_num_threads(),'dtype':str(short.dtype),'attention':m.config._attn_implementation,'seconds_128_tokens_one_loop':small_time,'estimated_two_four_loop_pass_seconds':estimate}),flush=True)
if estimate>150:
 write_json(P.parent/'results/CPU_REAL_REQUEST_PREFLIGHT.json',{'status':'not_run_estimate_over_limit','id':row['id'],'tokens':L,'threads':2,'attention':'sdpa','dtype':'bfloat16','estimate_seconds':estimate,'model_sha256':sha(PROJECT/'model/model.safetensors')});print('STOP estimated runtime over budget',flush=True);sys.exit(0)
scorer=OuroCachedScorer(m,4);tick=time.monotonic();_,historical,_=scorer._run(enc['input_ids'],positions=enc['predecessor_positions'],use_cache=False);execution=execute_request(m,enc);elapsed=time.monotonic()-tick;checks=[]
with tempfile.TemporaryDirectory(prefix='cpu_real_probe_',dir=PROJECT) as td:
 path=Path(td)/'x.safetensors';save_file(execution['tensors'],str(path));loaded=load_file(str(path));
 for k in range(4):
  ref=float(target_lp(m.lm_head(historical[k])[0].float(),ht).sum());z=loaded[f'loop{k+1}_lm_head_logits'];ts=torch.tensor(ht);saved=float(z.float().log_softmax(-1).gather(-1,ts[:,None]).sum());score=execution['scores'][k]
  if ref!=score or saved!=score:raise RuntimeError('score mismatch')
  checks.append({'loop':k+1,'historical':ref,'production':score,'saved':saved,'max_error':0.0})
write_json(P.parent/'results/CPU_REAL_REQUEST_PROBE.json',{'scope':'one real 25-shot CPU request, not GPU gate or benchmark','id':row['id'],'candidate':0,'tokens':L,'threads':2,'dtype':'bfloat16','attention':'sdpa','elapsed_seconds':elapsed,'checks':checks,'model_sha256':sha(PROJECT/'model/model.safetensors')});print('PASS real CPU historical/production/saved logits',elapsed,flush=True)
