"""PRE-REVIEW DRAFT: no GPU use until signed-off probe/gates and global lock."""
import sys,json,hashlib,os,argparse,subprocess,uuid,time,resource,psutil
from pathlib import Path
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=Path('/home/luansiyu/loop-latent');REPO=ROOT/'repo';EXP=REPO/'experiments/ouro_latent_state_mechanism_v1';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(EXP));sys.path.insert(0,str(ROOT/'research_plans/prefix_question_partition_draft_20261111'))
import torch,pyarrow.parquet as pq
from safetensors.torch import save_file,load_file
from transformers import AutoModelForCausalLM
from cohort_runner import bound_hashes,split_data_path,sha
from ledger_guard import require_production_ledger
from round2_intervention_runner import validate_envelope,canonical,write_once
from partition import partition
from pqr_core import run_question,NEW
from pqr_integrity import validate as validate_resume,validate_attempt,binding_guard,require_fresh_probe
from pqr_attempt import recheck
ORIG=ROOT/'orchestration_runs/run-20261107T053919Z-6d01138e';RESUME=ROOT/'orchestration_runs/resume-test-20261111';OLD=ROOT/'orchestration_runs/round2-full-20261111';BOUNDARY=ROOT/'research_plans/prompt_boundary_audit_20261111/boundary_ledger.jsonl';SAVED=ROOT/'research_plans/prefix_question_partition_draft_20261111/saved_offset_coverage.json'

def runtime_quick():
 return {'interpreter':str(Path(sys.executable).resolve()),'interpreter_sha256':sha(Path(sys.executable).resolve()),'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'flash_sdp':torch.backends.cuda.flash_sdp_enabled(),'mem_efficient_sdp':torch.backends.cuda.mem_efficient_sdp_enabled(),'math_sdp':torch.backends.cuda.math_sdp_enabled()}

def watched_paths(frozen):
 paths={Path(p) for p in frozen if p.startswith('/') and Path(p).is_file()}
 for split,v in frozen['splits'].items():
  paths.update(ROOT/k for k in v['bound_hashes']);paths.add((RESUME if split=='test' else ORIG)/('coverage_'+split+'.json'));paths.add(OLD/('coverage_'+split+'.json'))
 return {str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}

def bindings():
 fixed=[HERE/'pqr_core.py',HERE/'pqr_runner.py',HERE/'pqr_coverage.py',HERE/'pqr_binding.py',HERE/'pqr_integrity.py',HERE/'pqr_controller.py',HERE/'pqr_supervision.py',HERE/'pqr_probe_guard.py',HERE/'pqr_attempt.py',HERE/'pqr_analysis.py',HERE/'pqr_metrics.py',BOUNDARY,SAVED,ROOT/'research_plans/ROUND_3_TRAIN_VALIDATION_INFORMED_PLAN.md',ROOT/'research_plans/ROUND_3_EXECUTION_TERMS.md',ROOT/'research_plans/ROUND_3_INTERPRETATION_BOUNDARIES_ADDENDUM.md',ROOT/'research_plans/prefix_question_partition_draft_20261111/partition.py']
 expected_fixed={str(BOUNDARY):'a77258ade36415efb17ca09a886417f9b265ce53feec1589431dff6062f4e2bc',str(SAVED):'5263cc8bf07900822b11954194b7e9fd60c3066c9f43d96baee092157445dba6',str(ROOT/'research_plans/ROUND_3_TRAIN_VALIDATION_INFORMED_PLAN.md'):'c77e95bcdffdc6f837634cb795f3a38c538f97e32a8b618ac38c4411e032709d',str(ROOT/'research_plans/ROUND_3_EXECUTION_TERMS.md'):'4f852cc4282bf21321389ff3a24f503a70786d5bf5fd86167346e3344e5ca2ba',str(ROOT/'research_plans/ROUND_3_INTERPRETATION_BOUNDARIES_ADDENDUM.md'):'a9aec349a2b58b1e6b6aeedb13910c3a8f7ac58129ced4c8924a03ca9f265633'};assert all(sha(Path(p))==h for p,h in expected_fixed.items());saved=json.loads(SAVED.read_text());assert saved['status']=='ALL_SAVED_OFFSET_PARTITIONS_PASS_INPUTS_ONLY' and saved['boundary_ledger_sha256']==expected_fixed[str(BOUNDARY)] and saved['partition_sha256']==sha(ROOT/'research_plans/prefix_question_partition_draft_20261111/partition.py')
 actual_freeze=sorted(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,timeout=60).splitlines());assert actual_freeze==sorted((EXP/'environment_pip_freeze.txt').read_text().splitlines());x={str(p):sha(p) for p in fixed};x['runtime']=runtime_quick();x['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=REPO,text=True).strip();x['splits']={}
 for split,n in [('train',1119),('validation',299),('test',1172)]:
  nat=(RESUME if split=='test' else ORIG)/('coverage_'+split+'.json');nc=json.loads(nat.read_text());require_production_ledger(nc);old=OLD/('coverage_'+split+'.json');oc=json.loads(old.read_text());assert oc['status']=='COMPLETE_CUDA_REPLAY_ZERO_TOLERANCE' and oc['expected_questions']==oc['complete_questions']==n and oc['replayed_condition_logits']==11*n
  assert len({q['id'] for q in nc['questions']})==len({q['id'] for q in oc['questions']})==n and {q['id'] for q in nc['questions']}=={q['id'] for q in oc['questions']}
  bh=bound_hashes(split);assert all(oc['source'][k]==x['runtime'][k] for k in ('torch','cuda','gpu'));assert oc['source']['commit']==x['git_commit'] and oc['source']['bound_hashes']==bh and oc['source']['coverage_sha256']==sha(nat) and saved['stats'][split]['n']==n and saved['stats'][split]['natural_coverage_sha256']==sha(nat)
  x['splits'][split]={'natural_coverage_sha256':sha(nat),'old_coverage_sha256':sha(old),'bound_hashes':bh,'n':n}
 return x

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=('train','validation','test'),required=True);ap.add_argument('--out',required=True);ap.add_argument('--binding',required=True);ap.add_argument('--probe-id');ap.add_argument('--probe-receipt');ap.add_argument('--probe-shards');a=ap.parse_args();out=Path(a.out).resolve();out.relative_to(ROOT.resolve());out.mkdir(parents=True,exist_ok=True)
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info()[0]<=20*1024**3:raise RuntimeError('GPU admission')
 binding_path=Path(a.binding).resolve();binding_sha=sha(binding_path);frozen=json.loads(binding_path.read_text());assert sha(binding_path)==binding_sha;assert frozen['status']=='PQR_BINDING_COMPLETE_PENDING_PROBE' and frozen['binding']==bindings()
 if a.probe_id:
  assert frozen['longest_request']['id']==a.probe_id and frozen['longest_request']['split']==a.split
 else:
  if not a.probe_receipt or not a.probe_shards:raise RuntimeError('full requires independent probe receipt and shards')
  attempt_id,recheck_path,recheck_sha=recheck(out,binding_path,binding_sha,a.probe_receipt,a.probe_shards,frozen['longest_request'],HERE/'pqr_coverage.py',sha)
 if a.probe_id:attempt_id=uuid.uuid4().hex;recheck_path=None;recheck_sha=None
 split=a.split;nv=json.loads(((RESUME if split=='test' else ORIG)/('coverage_'+split+'.json')).read_text());ov=json.loads((OLD/('coverage_'+split+'.json')).read_text());natural={q['id']:q for q in nv['questions']};old={q['id']:q for q in ov['questions']};ledger={}
 for line in BOUNDARY.open():
  row=json.loads(line)
  if row['split']==split:assert row['id'] not in ledger;ledger[row['id']]=row
 data=pq.read_table(split_data_path(split)).to_pylist();assert len(data)==len(old)==len(natural)==len(ledger) and {r['id'] for r in data}==set(old)==set(natural)==set(ledger)
 if a.probe_id:data=[r for r in data if r['id']==a.probe_id];assert len(data)==1
 keys={hashlib.sha256(r['id'].encode()).hexdigest() for r in data};metas={p.stem for p in out.glob('*.json') if p.name!='_receipt.json'};tensors={p.stem for p in out.glob('*.safetensors')};assert not list(out.glob('*.partial')) and metas==tensors and metas<=keys and not (out/'_receipt.json').exists(),'partial/orphan/receipt/collision requires review';require_fresh_probe(bool(a.probe_id),metas);assert {p.name for p in out.iterdir()}=={k+'.json' for k in metas}|{k+'.safetensors' for k in tensors}
 torch.cuda.synchronize();pre_model_free,device_total=torch.cuda.mem_get_info();torch.cuda.reset_peak_memory_stats();started=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;assert model.config._attn_implementation=='sdpa' and next(model.parameters()).dtype==torch.bfloat16
 torch.cuda.synchronize();baseline_allocated=torch.cuda.memory_allocated();baseline_reserved=torch.cuda.memory_reserved();max_question_bytes=0;newly_computed=0
 runtime_snapshot=runtime_quick();assert runtime_snapshot==frozen['binding']['runtime'];watched=watched_paths(frozen['binding'])
 def quick_guard():
  assert sha(binding_path)==binding_sha and runtime_quick()==runtime_snapshot and all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in watched.items())
 for row in data:
  quick_guard();assert torch.cuda.mem_get_info()[0]>=4*1024**3,'GPU free reserve';ident=row['id'];key=hashlib.sha256(ident.encode()).hexdigest();nf=ORIG/'shards'/split/(key+'.json');assert sha(nf)==natural[ident]['metadata_sha256'];raw=nf.read_bytes();meta=json.loads(raw);regions=partition(raw,natural[ident]['metadata_sha256'],ledger[ident]);records=meta['records']
  oldf=OLD/split/(key+'.json');oldt=oldf.with_suffix('.safetensors');assert sha(oldf)==old[ident]['metadata_sha256'] and sha(oldt)==old[ident]['logits_sha256'];j=json.loads(oldf.read_text());assert j['payload_sha256']==hashlib.sha256(canonical(j['payload'])).hexdigest();identity=j['payload']['identity'];targets=[r['encoding']['target_ids'][0] for r in sorted(records,key=lambda r:r['candidate'])];validate_envelope(oldf,identity,meta['gold_index'],targets,records)
  assert identity['id']==ident and identity['split']==split and identity['metadata_sha256']==sha(nf) and {k:identity[k] for k in ov['source']}==ov['source']
  expected={'id':ident,'split':split,'binding_sha256':binding_sha,'old_metadata_sha256':sha(oldf),'old_logits_sha256':sha(oldt),'natural_metadata_sha256':sha(nf)};existing=out/(key+'.json')
  if existing.exists():
   prior=validate_resume(existing,canonical,sha,expected,j['payload']['result'],meta['gold_index'],targets,regions);validate_attempt(prior['payload'],binding_sha,ROOT,frozen['longest_request'],bool(a.probe_id),sha);max_question_bytes=max(max_question_bytes,existing.stat().st_size+existing.with_suffix('.safetensors').stat().st_size);assert max_question_bytes<=3*1024**2;quick_guard();print('PQR_RESUME_VALIDATED',split,ident,flush=True);continue
  result,logits=run_question(model,records,meta['gold_index'],regions,j['payload']['result'],load_file(str(oldt)));newly_computed+=1;assert set(logits)=={'corner_'+n for n in NEW}
  tf=out/(key+'.safetensors');tmp=tf.with_suffix('.safetensors.partial');save_file(logits,str(tmp));reloaded=load_file(str(tmp));assert set(reloaded)==set(logits) and all(torch.equal(reloaded[k],logits[k]) for k in logits);os.link(tmp,tf);tmp.unlink();payload={**expected,'attempt_id':attempt_id,'probe_recheck_path':recheck_path,'probe_recheck_sha256':recheck_sha,'result':result,'logits_sha256':sha(tf)};write_once(out/(key+'.json'),{'payload':payload,'payload_sha256':hashlib.sha256(canonical(payload)).hexdigest()});now=validate_resume(out/(key+'.json'),canonical,sha,expected,j['payload']['result'],meta['gold_index'],targets,regions);validate_attempt(now['payload'],binding_sha,ROOT,frozen['longest_request'],bool(a.probe_id),sha);max_question_bytes=max(max_question_bytes,tf.stat().st_size+(out/(key+'.json')).stat().st_size);assert max_question_bytes<=3*1024**2,'question storage bound';quick_guard();print('PQR_COMPLETE',split,ident,flush=True)
 if a.probe_id and newly_computed!=1:raise RuntimeError('probe must contain one freshly computed full question')
 torch.cuda.synchronize();measurement={'measurement_scope':'model load through fresh full-question forward and saved-output validation; RSS whole process','newly_computed_questions':newly_computed,'elapsed_seconds':time.perf_counter()-started,'pre_model_free_bytes':pre_model_free,'device_total_bytes':device_total,'baseline_allocated_bytes':baseline_allocated,'baseline_reserved_bytes':baseline_reserved,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'rss_bytes':psutil.Process().memory_info().rss,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'max_new_question_bytes':max_question_bytes}
 assert sha(binding_path)==binding_sha and bindings()==frozen['binding'];write_once(out/'_receipt.json',{'status':'PQR_PROBE_PENDING_REPLAY' if a.probe_id else 'PQR_SPLIT_PENDING_COVERAGE','split':split,'questions':len(data),'binding_sha256':binding_sha,'attempt_id':attempt_id,'probe_recheck_path':recheck_path,'probe_recheck_sha256':recheck_sha,'resources':measurement})
if __name__=='__main__':main()
