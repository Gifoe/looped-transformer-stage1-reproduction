"""Shard native BF16 activations per question; refuses to run without parity gate."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file,load_file
from transformers import AutoTokenizer,AutoModelForCausalLM
from prompt_spans import make_question,token_spans
from native_capture import capture,candidate_logprob
from manifest_guard import validate_manifest
P=Path(__file__).resolve();REPO=P.parents[2];PROJECT=REPO.parent
sys.path.insert(0,str(REPO/'experiments/ouro_weight_space_repair_v2/scripts'))
from r10_core import question_prompt,joint_prefix

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_json(path,value):
 tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)
def write_new_json(path,value):
 tmp=path.with_suffix(path.suffix+'.partial')
 with tmp.open('x',encoding='utf8') as f:
  f.write(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 try:os.link(tmp,path,follow_symlinks=False)
 finally:tmp.unlink()
def complete_shard(meta_path,tensor_path,row_id,hashes):
 if not meta_path.exists(): return False
 meta=json.loads(meta_path.read_text(encoding='utf-8'))
 if meta['id']!=row_id or meta['sha256']!=hashes or not tensor_path.exists() or sha(tensor_path)!=meta['tensor_sha256']:
  raise RuntimeError('corrupt or stale shard: '+row_id)
 return True
def split_data_path(split):
 if split not in ('train','validation','test'): raise ValueError('unknown split')
 return PROJECT/'data'/f'{split}.parquet'

def bound_hashes(split):
 dependencies=[P,P.with_name('prompt_spans.py'),P.with_name('native_capture.py'),REPO/'experiments/ouro_weight_space_repair_v2/scripts/r10_core.py',split_data_path(split),PROJECT/'model/model.safetensors',REPO/'experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json']
 dependencies += sorted((PROJECT/'model').glob('*.json')) + sorted((PROJECT/'model').glob('*.py')) + [PROJECT/'model/merges.txt', P.with_name('environment_pip_freeze.txt'), P.with_name('environment_conda_explicit.txt'), REPO/'experiments/ouro_weight_space_repair_v2/scripts/r7_core.py',REPO/'experiments/ouro_weight_space_repair_v2/scripts/ouro_cached_scoring.py',REPO/'experiments/ouro_weight_space_repair_v2/scripts/r7_inference.py',REPO/'experiments/ouro_weight_space_repair_v2/scripts/r7_math.py',PROJECT/'conda-env/lib/python3.11/site-packages/lm_eval/api/model.py',REPO/'experiments/ouro_weight_space_repair_v2/scripts/runtime.py',split_data_path('validation'),P.with_name('audit_input_manifest.py'),P.with_name('manifest_guard.py'),P.parent/'results/INPUT_MANIFEST_SUMMARY.json'] + [P.parent/'results'/f'INPUT_MANIFEST_{name}.jsonl' for name in ('train','validation','test')]
 hashes={str(p.relative_to(PROJECT)):sha(p) for p in dependencies}
 return hashes
def execute_request(model,request):
 positions=sorted(set(sum(request['spans'].values(),[])+request['predecessor_positions']))
 if len(request['input_ids'])>model.config.max_position_embeddings:raise ValueError('input exceeds model context')
 x=torch.tensor([request['input_ids']],dtype=torch.long,device=model.lm_head.weight.device)
 with capture(model,positions,4) as states:model(x,labels=None,use_cache=False,exit_at_step=3)
 scores=[];tensors={};indices=[positions.index(z) for z in request['predecessor_positions']]
 for k,state in enumerate(states['loop_post_norm']):
  tensors[f'loop{k+1}_post_norm']=state.contiguous()
  logits=model.lm_head(state[indices].unsqueeze(0).to(model.lm_head.weight.device))[0]
  tensors[f'loop{k+1}_lm_head_logits']=logits.detach().cpu().contiguous()
  target=torch.tensor(request['target_ids'],device=logits.device,dtype=torch.long)
  scores.append(float(logits.float().log_softmax(-1).gather(-1,target[:,None]).sum()))
 for name,value in states['layer_states'].items():tensors[name]=value.contiguous()
 return {'tensors':tensors,'scores':scores,'positions':positions,'events':states['events']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',required=True,choices=['train','validation','test']);ap.add_argument('--gate',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
 for name in {args.split,'validation'}:
  validate_manifest(name,split_data_path(name),P.parent/'results/INPUT_MANIFEST_SUMMARY.json',P.parent/'results'/f'INPUT_MANIFEST_{name}.jsonl')
 gate=json.loads(Path(args.gate).read_text(encoding='utf-8'))
 # A gate is a separately audited artifact; it must bind the exact code and inputs.
 required=('historical_scorer','independent_depths','identity_hook','bf16_roundtrip','repeatability','single_sequence','causal_mask')
 if gate.get('status')!='passed' or any(gate.get('checks',{}).get(k)!='passed' for k in required):raise RuntimeError('complete numerical parity gate required')
 hashes=bound_hashes(args.split)
 if gate.get('sha256')!=hashes:raise RuntimeError('parity gate does not bind current code/data/model')
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
 evidence=(PROJECT/gate['evidence_path']).resolve();evidence.relative_to(PROJECT.resolve())
 if gate.get('producer_sha256')!=sha(P.with_name('gpu_parity_gate.py')) or not evidence.is_file() or sha(evidence)!=gate.get('evidence_sha256'):raise RuntimeError('gate evidence/provenance mismatch')
 if gate.get('gpu')!=torch.cuda.get_device_name() or gate.get('torch')!=torch.__version__ or gate.get('cuda')!=torch.version.cuda:raise RuntimeError('gate runtime mismatch')
 evidence_data=json.loads(evidence.read_text(encoding='utf-8'))
 if evidence_data.get('gpu')!=gate['gpu'] or not evidence_data.get('records') or evidence_data.get('max_allowed_error')!=0.0:raise RuntimeError('invalid gate evidence')
 out=Path(args.out).resolve();out.relative_to(PROJECT.resolve());out.mkdir(parents=True,exist_ok=True)
 t=AutoTokenizer.from_pretrained(PROJECT/'model',local_files_only=True,trust_remote_code=True)
 m=AutoModelForCausalLM.from_pretrained(PROJECT/'model',trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
 m.config.use_cache=False;m.model.config.use_cache=False
 demo=json.loads((REPO/'experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json').read_text(encoding='utf-8'));prefix=joint_prefix(demo);rows=pq.read_table(split_data_path(args.split)).to_pylist()
 if len({row['id'] for row in rows})!=len(rows):raise RuntimeError('duplicate question IDs')
 for row in rows:
  name=hashlib.sha256(row['id'].encode()).hexdigest();meta_path=out/(name+'.json');tensor_path=out/(name+'.safetensors')
  if complete_shard(meta_path,tensor_path,row['id'],hashes):continue
  prompt,spans=make_question(prefix,row['question'],row['choices']['text'],question_prompt);tensors={};records=[]
  with torch.inference_mode():
   for i in range(len(row['choices']['text'])):
    request=token_spans(t,prompt,' '+chr(65+i),spans)
    execution=execute_request(m,request);positions=execution['positions'];scores=execution['scores']
    for name,value in execution['tensors'].items():tensors[f'candidate{i}_{name}']=value
    records.append({'candidate':i,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'mask_semantics':'argument omitted (native unpadded batch of one)','attention_mask_argument':None,'position_ids_argument':None,'effective_position_ids':list(range(len(request['input_ids']))),'batch_size':1,'input_ids_sha256':hashlib.sha256(json.dumps(request['input_ids']).encode()).hexdigest(),'request':{'context':prompt,'continuation':' '+chr(65+i)},'encoding':request,'captured_positions':positions,'scores':scores,'event_sequence':execution['events']})
  temp=tensor_path.with_suffix('.safetensors.partial');save_file(tensors,str(temp));reloaded=load_file(str(temp))
  if set(reloaded)!=set(tensors) or any(not torch.equal(reloaded[k],tensors[k]) for k in tensors):raise RuntimeError('BF16 shard roundtrip failed')
  os.link(temp,tensor_path,follow_symlinks=False);temp.unlink()
  gold=row['choices']['label'].index(row['answerKey'])
  rankings=[sorted(range(len(records)),key=lambda j:-records[j]['scores'][k]) for k in range(4)]
  predictions=[order[0] for order in rankings]
  margins=[records[gold]['scores'][k]-max(records[j]['scores'][k] for j in range(len(records)) if j!=gold) for k in range(4)]
  write_new_json(meta_path,{'id':row['id'],'split':args.split,'gold_index':gold,'rankings':rankings,'predictions':predictions,'gold_margins':margins,'sha256':hashes,'tensor_sha256':sha(tensor_path),'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'attention':'sdpa','dtype':'bfloat16','use_cache':False},'records':records})
if __name__=='__main__':main()
