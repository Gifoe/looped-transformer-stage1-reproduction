"""ROUND 2 source-bound per-question runner. Start only under supervised task lock."""
import argparse,hashlib,json,os,sys,time,subprocess,math
from pathlib import Path
if not __debug__:raise RuntimeError('optimized Python forbidden')
import torch,pyarrow.parquet as pq,psutil
from safetensors.torch import save_file,load_file
from transformers import AutoModelForCausalLM
from cohort_runner import bound_hashes,sha,split_data_path
from ledger_guard import require_production_ledger
from round2_intervention_core import run_question,CORE,NORM
P=Path(__file__).resolve();ROOT=P.parents[2].parent;ORIG=ROOT/'orchestration_runs/run-20261107T053919Z-6d01138e';RESUME=ROOT/'orchestration_runs/resume-test-20261111';ANCHOR=ROOT/'orchestration_runs/round2-anchor-20261111'
def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def write_once(path,obj):
 tmp=path.with_suffix(path.suffix+'.partial')
 with tmp.open('xb') as f:f.write(canonical(obj)+b'\n');f.flush();os.fsync(f.fileno())
 os.link(tmp,path);tmp.unlink()
def validate_envelope(path,identity,expected_gold,expected_targets,original_records):
 j=json.loads(path.read_text());payload=j.get('payload');assert isinstance(payload,dict) and j.get('payload_sha256')==hashlib.sha256(canonical(payload)).hexdigest()
 assert payload['identity']==identity;r=payload['result'];n=len(expected_targets);assert r['gold']==expected_gold and r['target_ids']==expected_targets and n>=2
 records=sorted(original_records,key=lambda z:z['candidate']);assert len(records)==n and [x['candidate'] for x in records]==list(range(n))
 ids=records[0]['encoding']['input_ids'];pos=records[0]['encoding']['predecessor_positions'];assert len(pos)==1 and all(x['encoding']['input_ids']==ids and x['encoding']['predecessor_positions']==pos and x['encoding']['target_ids']==[expected_targets[i]] for i,x in enumerate(records))
 assert r['input_length']==len(ids) and r['readout_positions']==pos and r['context_positions_count']==len(ids)-len(pos)
 assert r['baseline_K1']['scores']==[x['scores'][0] for x in records] and r['baseline_K2']['scores']==[x['scores'][1] for x in records]
 assert set(r['conditions'])==set(CORE) and set(r['norm_conditions'])==set(NORM)
 assert r['controls']=={'K2_identity':'exact','K1_full_anchor':'exact','last_layer_C_only':'exact','rescale_K2_to_K2':'exact'}
 assert set(r['norms'])=={'K1DIR_K2NORM','K2DIR_K1NORM'} and len(r['readout_positions'])==1 and r['context_positions']==[i for i in range(r['input_length']) if i not in set(r['readout_positions'])]
 for name in r['norms']:
  z=r['norms'][name];assert set(z)=={'source_norm','reference_norm','actual_bf16_norm','max_abs_actual_minus_reference'} and math.isfinite(z['max_abs_actual_minus_reference'])
  for k in ('source_norm','reference_norm','actual_bf16_norm'):assert len(z[k])==1 and len(z[k][0])==r['input_length'] and all(math.isfinite(x) and x>=0 for x in z[k][0])
 wrong=r['k1_highest_wrong'];assert isinstance(wrong,int) and 0<=wrong<n and wrong!=expected_gold
 groups=[r['baseline_K1'],r['baseline_K2']]+[r['conditions'][x] for x in CORE]+[r['norm_conditions'][x] for x in NORM]
 for x in groups:
  scores=x['scores'];assert len(scores)==n and all(isinstance(v,(int,float)) and math.isfinite(v) for v in scores);prediction=max(range(n),key=lambda i:(scores[i],-i));assert x['prediction']==prediction and x['top_tie']==(sum(v==scores[prediction] for v in scores)>1) and x['gold_vs_fixed_wrong_gap']==scores[expected_gold]-scores[wrong]
 assert wrong==max((i for i in range(n) if i!=expected_gold),key=lambda i:(r['baseline_K1']['scores'][i],-i))
 tensor=path.with_suffix('.safetensors');assert tensor.is_file() and sha(tensor)==payload['logits_sha256'];values=load_file(str(tensor))
 keys={'baseline_K1','baseline_K2','negative_last_C'}|{'core_'+x for x in CORE}|{'norm_'+x for x in NORM};assert set(values)==keys
 assert all(v.dtype==torch.bfloat16 and tuple(v.shape)==(1,49152) and torch.isfinite(v).all() for v in values.values())
 assert torch.equal(values['baseline_K1'],values['core_K1C_K1R']) and torch.equal(values['baseline_K2'],values['core_K2C_K2R']) and torch.equal(values['baseline_K2'],values['negative_last_C'])
 return j
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',required=True,choices=('train','validation','test'));ap.add_argument('--out',required=True);ap.add_argument('--probe-id');a=ap.parse_args()
 out=Path(a.out).resolve();out.relative_to(ROOT.resolve());out.mkdir(parents=True,exist_ok=True)
 if list(out.glob('*.partial')):raise RuntimeError('incomplete partial output requires review')
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
 free,total=torch.cuda.mem_get_info();assert free>20*1024**3
 anchor=json.loads((ANCHOR/'anchor.json').read_text());post=json.loads((ANCHOR/'anchor_post_audit.json').read_text());assert anchor['status']=='ANCHORS_PASSED_ONLY' and post['status']=='POST_AUDIT_PASSED' and post['anchor_sha256']==sha(ANCHOR/'anchor.json')
 assert anchor['source_sha256']['hook']==sha(P.with_name('round2_full_sequence.py'))
 root=RESUME if a.split=='test' else ORIG;coverage_path=root/f'coverage_{a.split}.json';coverage=json.loads(coverage_path.read_text());require_production_ledger(coverage);ledger={q['id']:q for q in coverage['questions']};assert len(ledger)==coverage['counts']['expected_questions']==coverage['counts']['complete_questions']
 gate=json.loads((ORIG/'gates'/f'GPU_GATE_{a.split}.json').read_text());binding=bound_hashes(a.split);assert gate['status']=='passed' and gate['sha256']==binding
 frozen=sorted((P.parent/'environment_pip_freeze.txt').read_text().splitlines());current=sorted(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,timeout=60).splitlines());assert current==frozen
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parents[2],text=True).strip();assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=P.parents[2],text=True).strip()
 source={str(p):sha(p) for p in (P,P.with_name('round2_intervention_core.py'),P.with_name('round2_full_sequence.py'),P.with_name('ledger_guard.py'),P.with_name('cohort_runner.py'),P.with_name('round2_intervention_coverage.py'))};base_id={'commit':commit,'source_sha256':source,'bound_hashes':binding,'coverage_sha256':sha(coverage_path),'anchor_sha256':sha(ANCHOR/'anchor.json'),'anchor_post_audit_sha256':sha(ANCHOR/'anchor_post_audit.json'),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda}
 data=pq.read_table(split_data_path(a.split)).to_pylist();assert len(data)==len(ledger) and {r['id'] for r in data}==set(ledger);demo=json.loads((P.parents[2]/'experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json').read_text());demos=set(demo['ids']);assert len(demos)==25
 if a.probe_id:
  data=[r for r in data if r['id']==a.probe_id];assert len(data)==1
 keys={hashlib.sha256(r['id'].encode()).hexdigest() for r in data};metas={p.stem for p in out.glob('*.json') if p.name!='_receipt.json'};tensors={p.stem for p in out.glob('*.safetensors')}
 if (out/'_receipt.json').exists() or metas!=tensors or not metas<=keys or any(p.name not in {k+'.json' for k in keys}|{k+'.safetensors' for k in keys} for p in out.iterdir()):raise RuntimeError('unexpected output, receipt, partial or orphan')
 torch.cuda.reset_peak_memory_stats();model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False
 assert not model.training and model.config._attn_implementation=='sdpa' and next(model.parameters()).dtype==torch.bfloat16
 baseline_mem=torch.cuda.memory_allocated();proc=psutil.Process();start=time.perf_counter();done=0
 for row in data:
  ident=row['id'];q=ledger[ident];assert q['status']=='complete';key=hashlib.sha256(ident.encode()).hexdigest();meta_path=ORIG/'shards'/a.split/(key+'.json');tensor_path=meta_path.with_suffix('.safetensors');metahash=sha(meta_path);tensorhash=sha(tensor_path);assert metahash==q['metadata_sha256'] and tensorhash==q['tensor_sha256'];meta=json.loads(meta_path.read_text());assert meta['id']==ident and meta['sha256']==binding
  records=meta['records'];idhash=hashlib.sha256(json.dumps(records[0]['encoding']['input_ids']).encode()).hexdigest();assert all(r['input_ids_sha256']==idhash for r in records)
  identity={**base_id,'split':a.split,'id':ident,'metadata_path':str(meta_path),'metadata_sha256':metahash,'tensor_path':str(tensor_path),'tensor_sha256':tensorhash,'input_ids_sha256':idhash,'is_demonstration':ident in demos}
  path=out/(key+'.json')
  targets=[r['encoding']['target_ids'][0] for r in sorted(records,key=lambda z:z['candidate'])]
  if path.exists():validate_envelope(path,identity,meta['gold_index'],targets,records);done+=1;continue
  if torch.cuda.mem_get_info()[0]<4*1024**3:raise RuntimeError('GPU free memory below reserve')
  result,logits=run_question(model,records,meta['gold_index']);tensor_out=path.with_suffix('.safetensors');tmp=tensor_out.with_suffix('.safetensors.partial')
  if tensor_out.exists() or tmp.exists():raise RuntimeError('tensor output collision')
  save_file(logits,str(tmp));loaded=load_file(str(tmp));assert set(loaded)==set(logits) and all(torch.equal(loaded[k],logits[k]) for k in logits);os.link(tmp,tensor_out);tmp.unlink()
  payload={'identity':identity,'result':result,'logits_sha256':sha(tensor_out)};write_once(path,{'payload':payload,'payload_sha256':hashlib.sha256(canonical(payload)).hexdigest()});validate_envelope(path,identity,meta['gold_index'],targets,records);done+=1
  print('QUESTION_COMPLETE',a.split,done,'/',len(data),ident,'seconds',round(result['elapsed_seconds'],3),flush=True)
 if bound_hashes(a.split)!=binding or sha(coverage_path)!=base_id['coverage_sha256'] or any(sha(Path(p))!=h for p,h in source.items()):raise RuntimeError('source drift')
 if sorted(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,timeout=60).splitlines())!=frozen:raise RuntimeError('environment drift')
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parents[2],text=True).strip()!=commit:raise RuntimeError('commit drift')
 status={'status':'PROBE_COMPLETE_ONLY' if a.probe_id else 'SPLIT_COMPLETE_PENDING_COVERAGE','kind':'ROUND2_INTERVENTION','split':a.split,'questions':done,'expected':len(data),'elapsed_seconds':time.perf_counter()-start,'gpu_baseline_allocated':baseline_mem,'gpu_peak_allocated':torch.cuda.max_memory_allocated(),'gpu_peak_reserved':torch.cuda.max_memory_reserved(),'rss_bytes':proc.memory_info().rss,'source':base_id}
 write_once(out/'_receipt.json',status);print('RUN_COMPLETE',json.dumps(status),flush=True)
if __name__=='__main__':main()
