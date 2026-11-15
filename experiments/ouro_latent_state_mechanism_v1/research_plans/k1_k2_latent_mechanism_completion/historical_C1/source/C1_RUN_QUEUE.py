"""Frozen finite C1 execution queue; no outcome-based scientific branching."""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/ouro-k1-k2-mechanism')
PY='/root/autodl-tmp/ouro-e6a/venv/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.partial');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def event(status,**kw):
 row={'status':status,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'queue_pid':os.getpid(),**kw}
 with (ROOT/'C1_QUEUE_JOURNAL.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 write(ROOT/'C1_QUEUE_STATE.json',row);print(json.dumps(row),flush=True)

def main():
 ref=ROOT/'TRAIN_REFERENCE_TECHNICAL_DEPENDENCIES_v2_COMPLETE.json'
 tech=ROOT/'TRAIN_TREATMENT_TECHNICAL2_v2_COMPLETE.json'
 independent=ROOT/'independent_replay/technical_v1/COMPLETE.json'
 assert read(ref)['status']=='COMPLETE' and read(ref)['n']==4
 assert read(tech)['status']=='COMPLETE' and read(tech)['n']==2
 assert read(independent)['status']=='PASS' and len(read(independent)['cases'])==2
 for name,expected in [('C1_COLLECT_v2.py','0c905c295f8dd418addd7a4e27f393d928135c072307c7af06e0ade69fcdb6d9'),
                       ('branch_interfaces_v2.py','d52a12c55cddb179fbcfb384897867118bd5dd6bee22959aa211168e27c0d7ea')]:
  assert sha(ROOT/name)==expected
 frozen_names=['C1_COLLECT_v2.py','branch_interfaces_v2.py','C1_RUN_QUEUE.py','C1_CAUSAL_ANALYSIS.py',
     'C1_ALL_ENDPOINT_CPU_AUDIT.py','C1_AUDIT_TENSOR_IO.py','C1_INDEPENDENT_LAYER_REPLAY.py',
     'C1_INDEPENDENT_REPLAY_DRIVER.py','ASTRA_ORDER_C1_UPSTREAM_TO_L18.md','C1_FP32_NUMERIC_SELECTION.json',
     'inputs/COHORT_AND_OPERATOR_FREEZE.json','inputs/train.jsonl','inputs/validation.jsonl',
     'analysis/C1_ANALYSIS_SOURCE_FREEZE_v2.json']
 frozen={name:sha(ROOT/name) for name in frozen_names}
 freeze=ROOT/'C1_EXECUTION_FREEZE.json';assert not freeze.exists()
 write(freeze,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_hashes':frozen,
       'technical_reference_receipt_sha256':sha(ref),'technical_treatment_receipt_sha256':sha(tech),
       'independent_technical_receipt_sha256':sha(independent),
       'chronology':'Execution freeze after technical acceptance, before full cohort. Scientific order/cohort and original analysis source were frozen earlier. v1 failed scalar serialization and all chronology corrections are retained.',
       'permissions':'Fixed Train1094 then exposedVal299 mechanism study only; no Test or controller search.'})
 stages=[
  ('replay_fixture',[PY,str(ROOT/'C1_INDEPENDENT_REPLAY_DRIVER.py'),'--root',str(ROOT),'--selection','replay','--prepare-only']),
  ('train_reference',[PY,str(ROOT/'C1_COLLECT_v2.py'),'--stage','reference','--split','train','--batch','all']),
  ('train64_treatment',[PY,str(ROOT/'C1_COLLECT_v2.py'),'--stage','treatment','--split','train','--batch','train64']),
  ('train1030_treatment',[PY,str(ROOT/'C1_COLLECT_v2.py'),'--stage','treatment','--split','train','--batch','train1030']),
  ('train_independent_replay',[PY,str(ROOT/'C1_INDEPENDENT_REPLAY_DRIVER.py'),'--root',str(ROOT),'--selection','replay','--out',str(ROOT/'independent_replay/full_train_v1')]),
  ('validation_reference',[PY,str(ROOT/'C1_COLLECT_v2.py'),'--stage','reference','--split','validation','--batch','all']),
  ('validation_treatment',[PY,str(ROOT/'C1_COLLECT_v2.py'),'--stage','treatment','--split','validation','--batch','all']),
  ('train_cpu_endpoint_audit',[PY,str(ROOT/'C1_ALL_ENDPOINT_CPU_AUDIT.py'),'--root',str(ROOT),'--split','train']),
  ('validation_cpu_endpoint_audit',[PY,str(ROOT/'C1_ALL_ENDPOINT_CPU_AUDIT.py'),'--root',str(ROOT),'--split','validation']),
  ('train_analysis',[PY,str(ROOT/'C1_CAUSAL_ANALYSIS.py'),'--root',str(ROOT),'--phase','train']),
  ('validation_analysis',[PY,str(ROOT/'C1_CAUSAL_ANALYSIS.py'),'--root',str(ROOT),'--phase','validation'])]
 start=time.time();event('STARTED',execution_freeze_sha256=sha(freeze),stages=[s for s,_ in stages])
 try:
  for stage,cmd in stages:
   assert all(sha(ROOT/name)==expected for name,expected in frozen.items()),'Frozen source changed'
   logfile=ROOT/('queue_'+stage+'_v1.log');assert not logfile.exists()
   event('STAGE_STARTED',stage=stage,command=cmd,log=str(logfile))
   env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
   with logfile.open('x') as output:subprocess.run(cmd,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
   event('STAGE_COMPLETE',stage=stage,log_sha256=sha(logfile),elapsed_seconds=time.time()-start)
  event('COMPLETE',elapsed_seconds=time.time()-start,scope='BF16 fixed-cohort collection, independent replay and CPU analysis complete; FP32 numerical sensitivity and scientific interpretation still require completion.')
 except BaseException as exc:
  event('FAILED',stage=stage,exception=repr(exc),elapsed_seconds=time.time()-start)
  raise
if __name__=='__main__':main()
