"""Finite CPU-only required-control-audit queue; no outcome-based branching."""
import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/ouro-k1-k2-mechanism')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(status,**kw):
 x={'status':status,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),**kw}
 p=ROOT/'C1_CONTROL_AUDIT_QUEUE_STATE.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p);print(json.dumps(x),flush=True)
def main():
 freeze=ROOT/'analysis/C1_CONTROL_AUDIT_SOURCE_FREEZE.json'
 assert sha(freeze)=='d85dcdb832a817dafd9aca5009f1a9c4aaf5dec224f11c9f4615734da18d5b83'
 f=read(freeze);source=ROOT/'C1_CONTROL_TENSOR_ARITHMETIC_AUDIT.py'
 assert sha(source)==f['source_sha256'] and sha(ROOT/'C1_AUDIT_TENSOR_IO.py')==f['tensor_reader_sha256']
 try:
  for split,n in [('train',1094),('validation',299)]:
   gate=ROOT/(split.upper()+'_TREATMENTS_COMPLETE.json');state('WAITING_FOR_COMPLETE_SPLIT',split=split)
   while not gate.exists():
    if read(ROOT/'C1_QUEUE_STATE.json')['status']=='FAILED':raise RuntimeError('Producer failed before complete split')
    time.sleep(30)
   g=read(gate);assert g['status']=='COMPLETE' and g['n']==n
   assert g['source_identity']['collector_sha256']=='0c905c295f8dd418addd7a4e27f393d928135c072307c7af06e0ade69fcdb6d9'
   assert sha(source)==f['source_sha256'] and not (ROOT/'analysis'/(split.upper()+'_ALL_CONTROL_TENSOR_AUDIT.json')).exists()
   state('AUDITING_COMPLETE_SPLIT',split=split,producer_gate_sha256=sha(gate))
   with (ROOT/('control_audit_'+split+'_v1.log')).open('x') as log:
    subprocess.run(['/root/autodl-tmp/ouro-e6a/venv/bin/python','-u',str(source),'--root',str(ROOT),'--split',split],
        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
   assert read(ROOT/'analysis'/(split.upper()+'_ALL_CONTROL_TENSOR_AUDIT.json'))['status']=='PASS'
   state('SPLIT_PASS',split=split)
  state('COMPLETE')
 except BaseException as ex:state('FAILED',exception=repr(ex));raise
if __name__=='__main__':main()
