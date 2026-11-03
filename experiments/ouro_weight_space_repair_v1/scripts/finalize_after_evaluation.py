"""Bounded detached completion of reporting/audit after our own frozen worker."""
import json,os,subprocess,sys,time
from runtime import ROOT,save

def main():
 deadline=time.time()+5000;status=dict(status='waiting_for_frozen_evaluation',started_unix=time.time(),steps=[])
 save('logs/FINALIZATION_PIPELINE.json',status)
 try:
  while True:
   state=json.loads((ROOT/'logs/FROZEN_EVALUATION_PIPELINE.json').read_text())
   if state['status'] in ['complete','failed']:break
   if time.time()>deadline:raise RuntimeError('Own frozen evaluation did not finish in bounded finalization window')
   time.sleep(5)
  assert state['status']=='complete',state.get('error','Frozen evaluation failed')
  for name,args,timeout in [('report',['assemble_no_edit_report.py'],120),('audit',['audit_artifacts.py','--verify-model'],180),('package',['package_public_results.py'],180)]:
   step=dict(name=name,status='running',started_unix=time.time());status['steps'].append(step);status['status']='running';save('logs/FINALIZATION_PIPELINE.json',status)
   with (ROOT/'logs'/f'finalization_{name}.out').open('x') as out,(ROOT/'logs'/f'finalization_{name}.err').open('x') as err:
    p=subprocess.run([sys.executable,str(ROOT/'scripts'/args[0]),*args[1:]],cwd=ROOT,stdout=out,stderr=err,timeout=timeout)
   step.update(status='complete' if p.returncode==0 else 'failed',exit_code=p.returncode,finished_unix=time.time());save('logs/FINALIZATION_PIPELINE.json',status)
   if p.returncode:raise RuntimeError('Finalization failed at '+name)
  status.update(status='complete',finished_unix=time.time());save('logs/FINALIZATION_PIPELINE.json',status)
 except BaseException as exc:
  status.update(status='failed',error=repr(exc),finished_unix=time.time());save('logs/FINALIZATION_PIPELINE.json',status);raise
if __name__=='__main__':main()
