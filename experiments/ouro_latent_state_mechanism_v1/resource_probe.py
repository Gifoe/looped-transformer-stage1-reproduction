"""GPU peak and complete question shard probe on global input/position extremes."""
import argparse,json,sys,tempfile,shutil
from pathlib import Path
import pyarrow.parquet as pq
import torch,psutil
from safetensors.torch import save_file,load_file
from transformers import AutoTokenizer,AutoModelForCausalLM
from cohort_runner import write_new_json
from cohort_runner import PROJECT,REPO,split_data_path,execute_request,write_json
from prompt_spans import make_question,token_spans
from manifest_guard import validate_manifest
sys.path.insert(0,str(REPO/'experiments/ouro_weight_space_repair_v2/scripts'))
from r10_core import joint_prefix,question_prompt
P=Path(__file__).resolve();parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(P.parent/'results/GPU_RESOURCE_PROBE.json'));args=parser.parse_args();output=Path(args.out).resolve();output.relative_to(PROJECT.resolve())
if output.exists():raise RuntimeError('refusing to overwrite resource probe output')
records=[]
for split in ('train','validation','test'):
 validate_manifest(split,split_data_path(split),P.parent/'results/INPUT_MANIFEST_SUMMARY.json',P.parent/'results'/f'INPUT_MANIFEST_{split}.jsonl')
 with (P.parent/'results'/f'INPUT_MANIFEST_{split}.jsonl').open() as f:records.extend(json.loads(line) for line in f)
grouped={}
for r in records:grouped.setdefault((r['split'],r['id']),[]).append(r)
longest=max(records,key=lambda x:(x['input_length'],x['split'],x['id'],x['candidate']))
max_group=max(grouped.values(),key=lambda rs:(sum(x['selected_token_count'] for x in rs),rs[0]['split'],rs[0]['id']))
chosen={'longest_input':longest,'largest_question_capture':max_group[0]}
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 unavailable')
free,total=torch.cuda.mem_get_info()
if free<40*1024**3 or free/total<0.90:raise RuntimeError('GPU not sufficiently free for non-interfering probe')
model=AutoModelForCausalLM.from_pretrained(PROJECT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False
tok=AutoTokenizer.from_pretrained(PROJECT/'model',local_files_only=True,trust_remote_code=True);prefix=joint_prefix(json.loads((REPO/'experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json').read_text()));proc=psutil.Process();result={'pre_model_free_bytes':free,'device_total_bytes':total,'gpu':torch.cuda.get_device_name(),'rss_note':'sampled after candidate and serialization steps, not continuous peak','cases':{}}
with torch.inference_mode():
 for name,case in chosen.items():
  row=next(x for x in pq.read_table(split_data_path(case['split'])).to_pylist() if x['id']==case['id']);prompt,spans=make_question(prefix,row['question'],row['choices']['text'],question_prompt);torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();rss=[proc.memory_info().rss];disk=shutil.disk_usage(PROJECT).free;tensors={};candidate_meta=[]
  for j in range(len(row['choices']['text'])):
   enc=token_spans(tok,prompt,' '+chr(65+j),spans);run=execute_request(model,enc);tensors.update({f'candidate{j}_{k}':v for k,v in run['tensors'].items()});candidate_meta.append({'candidate':j,'selected_token_count':len(run['positions']),'input_length':len(enc['input_ids'])});rss.append(proc.memory_info().rss)
  with tempfile.TemporaryDirectory(prefix='resource_probe_',dir=PROJECT) as td:
   path=Path(td)/'states.safetensors';save_file(tensors,str(path));rss.append(proc.memory_info().rss);size=path.stat().st_size;loaded=load_file(str(path));rss.append(proc.memory_info().rss)
   if set(loaded)!=set(tensors) or any(not torch.equal(loaded[k],tensors[k]) for k in tensors):raise RuntimeError('roundtrip failed')
   result['cases'][name]={'id':case['id'],'split':case['split'],'candidate_count':len(candidate_meta),'candidates':candidate_meta,'baseline_allocated_bytes':baseline,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'sampled_peak_rss_bytes':max(rss),'serialized_bytes':size,'disk_free_before_bytes':disk,'disk_free_during_bytes':shutil.disk_usage(PROJECT).free}
  del tensors,loaded,run
write_new_json(output,result)
