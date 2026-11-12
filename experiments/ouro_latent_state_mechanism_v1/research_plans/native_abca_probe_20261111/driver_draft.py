"""Draft orchestration. Not runnable: requires independent supervisor and review."""
import os,sys,json,fcntl,shutil,subprocess
from pathlib import Path
import torch
from input_binding import inputs,ROOT,sha
from capture_core import capture_one
from persistence import save_capture,compare_all,write_once
from replay import replay
from finalize import publish
from provenance import freeze,verify,pqr_complete,verify_replay_ledger
from transformers import AutoModelForCausalLM
PQR=ROOT/'research_plans/pqr_eight_corner_code_20261111'
sys.path.insert(0,str(PQR))
from pqr_runner import bindings,watched_paths,runtime_quick

def perform(out):
 """Internal draft only; never call without approved process supervision."""
 out=Path(out).resolve();out.relative_to(ROOT/'orchestration_runs');out.mkdir(exist_ok=False)
 fd=int(os.environ['ABCA_LOCK_FD'])
 if os.fstat(fd).st_ino!=os.stat(ROOT/'ouro_watch.lock',follow_symlinks=False).st_ino or os.fstat(fd).st_dev!=os.stat(ROOT/'ouro_watch.lock',follow_symlinks=False).st_dev:raise RuntimeError('inherited lock identity')
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  run=ROOT/'orchestration_runs/pqr-round3-20261111';binding=run/'binding.json';binding_sha=sha(binding)
  source=json.loads(binding.read_text())['binding']
  orig=ROOT/'orchestration_runs/run-20261107T053919Z-6d01138e';resume=ROOT/'orchestration_runs/resume-test-20261111';cohort={}
  for split,n in (('train',1119),('validation',299),('test',1172)):
   path=(resume if split=='test' else orig)/('coverage_'+split+'.json')
   if sha(path)!=source['splits'][split]['natural_coverage_sha256']:raise RuntimeError('natural coverage drift')
   q=json.loads(path.read_text())['questions'];cohort[split]={x['id'] for x in q}
   if len(q)!=n or len(cohort[split])!=n:raise RuntimeError('natural cohort')
  pqr_evidence=pqr_complete(run,binding_sha,cohort)
  if bindings()!=source:raise RuntimeError('frozen source/runtime drift')
  watched=watched_paths(source);runtime=runtime_quick();selected=inputs()
  here=Path(__file__).resolve().parent
  files=[here/n for n in ('observer.py','integrity.py','capture_core.py','input_binding.py','persistence.py','replay.py','provenance.py','finalize.py','driver_draft.py','supervisor_draft.py')]+[ROOT/'research_plans/PREFIX_NATIVE_ABCA_PLAN.md',ROOT/'research_plans/PREFIX_NATIVE_ABCA_SELECTION.json',ROOT/'research_plans/natural_prefix_identity_20261111/reference_prefix_ids.json',Path('/home/luansiyu/loop-latent/conda-env/lib/python3.11/site-packages/transformers/integrations/sdpa_attention.py'),binding,ROOT/'model/config.json',ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/ouro_watch.py',PQR/'pqr_supervision.py',here/'prelaunch_guard.py',here/'unit_history.py',here/'launcher_draft.py']
  manifest=freeze(files,{'pqr_binding':source,'pqr_evidence':pqr_evidence},runtime)
  mf=out/'source_manifest.json';write_once(mf,(json.dumps(manifest,sort_keys=True,allow_nan=False)+'\n').encode());manifest_sha=sha(mf)
  def guard():
   if sha(mf)!=manifest_sha or pqr_complete(run,binding_sha,cohort)!=pqr_evidence:raise RuntimeError('manifest/PQR drift')
   verify(manifest,files,{'pqr_binding':source,'pqr_evidence':pqr_evidence},runtime_quick())
  config=json.loads((ROOT/'model/config.json').read_text())
  dimensions={'hidden_size':config['hidden_size'],'head_dim':config['head_dim'],'q_heads':config['num_attention_heads'],'kv_heads':config['num_key_value_heads'],'dtype':'torch.bfloat16','device':'cuda:0','prefix_length':1839,'vocab_size':config['vocab_size']}
  if shutil.disk_usage(ROOT).free<20*1024**3 or torch.cuda.mem_get_info()[0]<=20*1024**3:raise RuntimeError('resource preflight')
  model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False
  replay_ledger={}
  for label,key in (('A1','A'),('B','B'),('C','C'),('A2','A')):
   guard()
   if runtime_quick()!=runtime or any((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=v for p,v in watched.items()):raise RuntimeError('source/runtime changed')
   if torch.cuda.mem_get_info()[0]<4*1024**3 or shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('resource reserve')
   entry=selected[key];obs,scores,logits=capture_one(model,entry,1839,dimensions)
   identity={k:v for k,v in entry.items() if k not in ('ids','scores')};save_capture(out,label,identity,obs,scores,logits,manifest_sha);replay_ledger[label]=replay(out,label,entry,manifest_sha,dimensions,'cuda');guard()
  guard();lp=out/'replay_ledger.json';write_once(lp,(json.dumps(replay_ledger,sort_keys=True,allow_nan=False)+'\n').encode());ledger_sha=sha(lp)
  for label,key in (('A1','A'),('B','B'),('C','C'),('A2','A')):
   if replay(out,label,selected[key],manifest_sha,dimensions,'cuda')!=replay_ledger[label]:raise RuntimeError('post-replay drift')
  if sha(lp)!=ledger_sha:raise RuntimeError('ledger drift')
  verify_replay_ledger(out,replay_ledger,'cuda');result=compare_all(out)
  for label in replay_ledger:
   if sha(out/(label+'.json'))!=replay_ledger[label]['receipt_sha256'] or sha(out/(label+'.safetensors'))!=replay_ledger[label]['tensor_sha256']:raise RuntimeError('comparison source drift')
  for key,value in result.items():
   if 'rope_' in key and not value['bitwise_equal']:raise RuntimeError('prefix rope differs')
  if bindings()!=source:raise RuntimeError('source drift after capture')
  publish(out,result,runtime,manifest_sha,ledger_sha,replay_ledger,guard)
 finally:pass  # supervisor owns and releases inherited flock after process-tree cleanup
if __name__=='__main__':raise SystemExit('DISABLED: requires supervisor, persistence audit and Astra review')
