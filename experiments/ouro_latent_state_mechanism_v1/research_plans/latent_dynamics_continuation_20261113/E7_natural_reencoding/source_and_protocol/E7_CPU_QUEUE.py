"""Observe the existing collector, verify full units, run CPU Train then frozen Val. Never launch GPU."""
import argparse,json,hashlib,os,time,datetime,subprocess,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);t=p.with_name(p.name+'.partial');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='/root/autodl-tmp/ouro-e7/reencoding');p.add_argument('--collector-pid',type=int,required=True);p.add_argument('--poll-seconds',type=float,default=30);a=p.parse_args();r=Path(a.root);out=r/'analysis';out.mkdir(exist_ok=True);cap=r/'full_capture';statefile=out/'CPU_QUEUE_STATE.json';expected={'train':{},'validation':{}};qids={'train':set(),'validation':set()};start=time.time()
 with (r/'E7_VARIANTS.jsonl').open() as f:
  for line in f:
   q=json.loads(line);split=q['split'];assert split in expected;name=split+'_'+hashlib.sha256(q['variant_id'].encode()).hexdigest()+'.json';assert name not in expected[split];expected[split][name]={'id':q['id'],'variant_id':q['variant_id'],'p':q['p'],'s':q['s'],'input_ids_sha256':q['input_ids_sha256'],'variant_row_sha256':canonical(q)};qids[split].add(q['id'])
 assert [len(expected[s]) for s in ['train','validation']]==[17504,4776] and [len(qids[s]) for s in ['train','validation']]==[1094,299];core=read(cap/'IDENTITY.json');freeze=read(out/'E7_ANALYSIS_SOURCE_FREEZE.json');assert core['collector_sha256']==freeze['hashes']['E7_collector'];assert sha(r/'E7_REENCODING_ANALYSIS.py')==freeze['hashes']['analysis_source']
 def process_identity():
  try:
   raw=Path('/proc/'+str(a.collector_pid)+'/stat').read_text().split();cmd=Path('/proc/'+str(a.collector_pid)+'/cmdline').read_bytes().replace(b'\0',b' ').decode();return {'start_tick':raw[21],'state':raw[2],'command':cmd}
  except (FileNotFoundError,ProcessLookupError):return None
 initial=process_identity()
 if initial is not None:assert 'E7_REENCODING_COLLECT.py' in initial['command'] and 'full_capture' in initial['command']
 verified={'train':{},'validation':{}};phase='train';state={}
 def snapshot(status,**more):
  nonlocal state
  state={'status':status,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'queue_pid':os.getpid(),'queue_source_sha256':sha(__file__),'analysis_source_freeze_sha256':sha(out/'E7_ANALYSIS_SOURCE_FREEZE.json'),'collector_pid':a.collector_pid,'initial_collector_identity':initial,'phase':phase,'verified_train_units':len(verified['train']),'expected_train_units':17504,'verified_validation_units':len(verified['validation']),'expected_validation_units':4776,'wall_seconds':time.time()-start,**more};write(statefile,state);print('E7_CPU_QUEUE',json.dumps(state),flush=True)
 try:
  snapshot('OBSERVING_EXISTING_COLLECTOR')
  while True:
   live=process_identity();alive=live is not None and live['state']!='Z' and (initial is None or live['start_tick']==initial['start_tick'])
   if live is not None and alive:assert 'E7_REENCODING_COLLECT.py' in live['command'] and 'full_capture' in live['command'],'Collector PID identity changed'
   global_receipt=read(cap/'COMPLETE.json') if (cap/'COMPLETE.json').exists() else None
   if global_receipt is not None:assert global_receipt['status']=='COMPLETE' and global_receipt['n']==22280 and global_receipt['identity']==core
   if not alive and global_receipt is None:raise RuntimeError('Actual collector process terminal and full22280 atomic COMPLETE absent; no GPU restart or partial-coverage conclusion')
   available={f.name for f in cap.glob('*.json')};unexpected=available-set(expected['train'])-set(expected['validation'])-{'IDENTITY.json','COMPLETE.json'};assert not unexpected,('Unexpected capture unit files',sorted(unexpected)[:3])
   # Before Train freeze, never parse any new Validation outcome JSON.
   for name in sorted(available.intersection(expected[phase])-set(verified[phase])):
    file=cap/name;raw=file.read_bytes();u=json.loads(raw);meta=expected[phase][name];assert u['status']=='COMPLETE' and u['identity']==core and u['weight_versions_unchanged'];assert u['unit_payload_sha256']==canonical({k:v for k,v in u.items() if k!='unit_payload_sha256'});assert all(u[k]==v for k,v in meta.items());assert u['production_actual_counts']=={'model_body_forwards':1,'head_calls':4,'first_layer_calls':4} and len(u['native'])==4;assert u['split']==phase
    if u['p']==u['s']==0:assert u['identity_E6_bridge']['status']=='PASS'
    verified[phase][name]=hashlib.sha256(raw).hexdigest()
   snapshot('COVERAGE_VERIFICATION',collector_alive=alive,global_complete_present=global_receipt is not None)
   if len(verified[phase])==len(expected[phase]):
    if phase=='validation' and global_receipt is None:time.sleep(a.poll_seconds);continue
    marker=out/(phase.upper()+'_COMPLETE.json')
    if not marker.exists():
     snapshot('RUNNING_CPU_ANALYSIS',collector_alive=alive)
     subprocess.run([sys.executable,str(r/'E7_REENCODING_ANALYSIS.py'),'--phase',phase,'--root',str(r)],check=True)
    assert read(marker)['status']=='COMPLETE';assert (out/'E7_TRAIN_FROZEN_DECISIONS.json').exists()
    if phase=='train':phase='validation';snapshot('TRAIN_FROZEN_NOW_ALLOW_VAL_OUTCOMES');continue
    assert global_receipt is not None;retained={Path(row['file']).name:row['sha256'] for row in global_receipt['records']};assert len(retained)==22280 and retained=={**verified['train'],**verified['validation']};snapshot('COMPLETE',full_capture_receipt_sha256=sha(cap/'COMPLETE.json'),train_decisions_sha256=sha(out/'E7_TRAIN_FROZEN_DECISIONS.json'));write(out/'CPU_QUEUE_COMPLETE.json',state);return
   time.sleep(a.poll_seconds)
 except BaseException as ex:
  snapshot('TERMINAL_FAILURE_NO_AUTOMATIC_GPU_RESTART',exception=repr(ex));write(out/'CPU_QUEUE_TERMINAL_FAILURE.json',state);raise
if __name__=='__main__':main()
