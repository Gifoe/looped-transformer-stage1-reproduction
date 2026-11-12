"""Replay one completed resource probe; never changes input artifacts."""
import os,fcntl,json,hashlib,subprocess,sys,time
from pathlib import Path
if not __debug__:raise RuntimeError('optimized Python forbidden')
import torch
from safetensors.torch import load_file
from cohort_runner import sha,bound_hashes
from round2_intervention_core import CORE,NORM
from round2_intervention_runner import validate_envelope
from ouro_watch import machine,GIB
P=Path(__file__).resolve();ROOT=P.parents[2].parent;run=ROOT/'orchestration_runs/round2-question-probe-20261111';out=run/'question';orig=ROOT/'orchestration_runs/run-20261107T053919Z-6d01138e';resume=ROOT/'orchestration_runs/resume-test-20261111'
lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR,0o600)
try:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);snap=machine();assert snap['gpu']['free_mib']>20*1024 and not snap['gpu']['compute_pids'] and snap['gpu']['util_percent']<10 and snap['disk_free_bytes']>20*GIB
 receipt=json.loads((out/'_receipt.json').read_text());assert receipt['status']=='PROBE_COMPLETE_ONLY' and receipt['questions']==1;files=[p for p in out.glob('*.json') if p.name!='_receipt.json'];assert len(files)==1
 j=json.loads(files[0].read_text());identity=j['payload']['identity'];assert identity['id']=='MSA_2012_8_30' and identity['split']=='test' and receipt['source']['commit']=='1df704b1386a415ea996b1df941bb994bd9ca2f7'
 source=identity['source_sha256'];assert source==receipt['source']['source_sha256']
 for path,h in source.items():
  p=Path(path);relative=p.relative_to(ROOT/'repo');raw=subprocess.check_output(['git','show',receipt['source']['commit']+':'+relative.as_posix()],cwd=ROOT/'repo');assert hashlib.sha256(raw).hexdigest()==h
 assert sha(P.with_name('round2_intervention_core.py'))==source[str(P.with_name('round2_intervention_core.py'))] and sha(P.with_name('round2_full_sequence.py'))==source[str(P.with_name('round2_full_sequence.py'))]
 assert identity['bound_hashes']==bound_hashes('test') and identity['coverage_sha256']==sha(resume/'coverage_test.json')
 key=hashlib.sha256(identity['id'].encode()).hexdigest();mp=orig/'shards/test'/(key+'.json');tp=mp.with_suffix('.safetensors');cov=json.loads((resume/'coverage_test.json').read_text());q=next(x for x in cov['questions'] if x['id']==identity['id']);assert sha(mp)==q['metadata_sha256']==identity['metadata_sha256'] and sha(tp)==q['tensor_sha256']==identity['tensor_sha256'];meta=json.loads(mp.read_text());records=sorted(meta['records'],key=lambda z:z['candidate']);targets=[r['encoding']['target_ids'][0] for r in records]
 j=validate_envelope(files[0],identity,meta['gold_index'],targets,records);result=j['payload']['result'];tensors=load_file(str(files[0].with_suffix('.safetensors')))
 mapping={'baseline_K1':result['baseline_K1']['scores'],'baseline_K2':result['baseline_K2']['scores'],'negative_last_C':result['baseline_K2']['scores']};mapping.update({'core_'+k:result['conditions'][k]['scores'] for k in CORE});mapping.update({'norm_'+k:result['norm_conditions'][k]['scores'] for k in NORM})
 for name,expected in mapping.items():
  snap=machine();assert not snap['gpu']['compute_pids'] or all(pid==os.getpid() for pid in snap['gpu']['compute_pids']);assert snap['disk_free_bytes']>20*GIB
  z=tensors[name].to('cuda');actual=[]
  for target in targets:
   t=torch.tensor([target],device='cuda',dtype=torch.long);actual.append(float(z.float().log_softmax(-1).gather(-1,t[:,None]).sum()))
  assert actual==expected,(name,actual,expected)
 report={'status':'PROBE_REPLAY_PASSED_ONLY','case':identity['id'],'groups':len(mapping),'score_comparisons':sum(len(v) for v in mapping.values()),'tolerance':0.0,'device':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'original_probe_commit':receipt['source']['commit'],'probe_receipt_sha256':sha(out/'_receipt.json'),'probe_record_sha256':sha(files[0]),'logits_sha256':sha(files[0].with_suffix('.safetensors')),'current_validation_sha256':sha(P.with_name('round2_intervention_runner.py')),'coverage_sha256':sha(resume/'coverage_test.json'),'note':'single question only; not cohort completion'}
 dest=run/'probe_replay_audit.json'
 with dest.open('x') as f:json.dump(report,f,indent=2);f.flush();os.fsync(f.fileno())
 print(json.dumps(report),flush=True)
finally:os.close(lock)
