"""Run the already frozen numerical8 once after fixed BF16 GPU collection."""
import datetime,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/ouro-k1-k2-mechanism')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(status,**kw):
 row={'status':status,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),**kw}
 p=ROOT/'C1_NUMERIC_QUEUE_STATE.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(row,indent=2)+'\n');os.replace(tmp,p)
 print(json.dumps(row),flush=True)
def main():
 source=ROOT/'C1_FP32_NUMERICAL.py';freeze=ROOT/'numeric_fp32/SOURCE_AND_SAMPLE_FREEZE.json'
 assert sha(source)=='93391406c417a51290bf60434eb64bbb3645e6f62130f333e3bde268aef22f87'
 expected_freeze='93ff491cda06f9063f2a9d1b4510aac6d3d896373f5baf2cf4f6407bf1cdf13b'
 assert sha(freeze)==expected_freeze
 state('WAITING_FOR_FIXED_BF16_COLLECTION',source_sha256=sha(source),freeze_sha256=sha(freeze))
 gate=ROOT/'VALIDATION_TREATMENTS_COMPLETE.json'
 while not gate.exists():
  if (ROOT/'C1_QUEUE_STATE.json').exists() and read(ROOT/'C1_QUEUE_STATE.json')['status']=='FAILED':
   state('STOPPED_ON_BF16_PRODUCER_FAILURE');raise RuntimeError('BF16 queue failed; no numerical GPU run')
  time.sleep(30)
 receipt=read(gate);assert receipt['status']=='COMPLETE' and receipt['n']==299
 assert receipt['source_identity']['collector_sha256']=='0c905c295f8dd418addd7a4e27f393d928135c072307c7af06e0ade69fcdb6d9'
 while True:
  lock=open('/root/autodl-tmp/ouro-e6a/gpu.lock','a+')
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:lock.close();time.sleep(5);continue
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close();break
 assert sha(source)=='93391406c417a51290bf60434eb64bbb3645e6f62130f333e3bde268aef22f87' and sha(freeze)==expected_freeze
 cmd=['/root/autodl-tmp/ouro-e6a/venv/bin/python','-u',str(source),'--phase','run']
 state('RUNNING_FIXED_NUMERICAL8',command=cmd,BF16_validation_gate_sha256=sha(gate))
 try:
  with (ROOT/'numeric_fp32_run_v1.log').open('x') as log:
   subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  assert read(ROOT/'numeric_fp32/COMPLETE.json')['n']==8
  state('COMPLETE',numeric_receipt_sha256=sha(ROOT/'numeric_fp32/COMPLETE.json'))
 except BaseException as exc:state('FAILED',exception=repr(exc));raise
if __name__=='__main__':main()
