"""One CPU180s budget covers solver worker, certificate analyzer and exits."""
import os,sys,time,signal,subprocess
from r31_oracle_common import ROOT,read,sha,save,gate,OUTPUTS
KILL=getattr(signal,'SIGKILL',9)
def main():
 s=gate();assert s['resources']['hard_seconds']==180;assert all(not(ROOT/p).exists() for p in OUTPUTS);old=s['inherited_GPU_ledgers'];assert len(old)==18 and all(sha(p)==h for p,h in old.items());total=sum(read(p)['used_seconds'] for p in old);assert total==20755.46525352029
 start=time.monotonic();wall=time.time();state=dict(status='starting',supervisor_pid=os.getpid(),started_unix=wall,hard_seconds=180,soft_seconds=175,kill_seconds=179,worker_exit_code=None);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',R31_CPU_DEADLINE_UNIX=str(wall+175),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1');process=None
 def stop(sig):
  if os.name=='nt':process.kill() if sig==KILL else process.terminate()
  else:os.killpg(process.pid,sig)
 def run(script,purpose):
  nonlocal process
  with (ROOT/('logs/r31_ORACLE_'+purpose+'.out')).open('x') as out,(ROOT/('logs/r31_ORACLE_'+purpose+'.err')).open('x') as err:
   kw=dict(start_new_session=True,preexec_fn=lambda:os.nice(15)) if os.name!='nt' else {}
   process=subprocess.Popen([sys.executable,'-B','-u',script],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,**kw);state[purpose+'_pid']=process.pid;save('logs/r31_ORACLE_SUPERVISOR.json',state,replace=True)
   try:return process.wait(timeout=max(0,start+175-time.monotonic()))
   except subprocess.TimeoutExpired:
    state['timed_out']=True;stop(signal.SIGTERM)
    try:return process.wait(timeout=max(0,start+179-time.monotonic()))
    except subprocess.TimeoutExpired:stop(KILL);return process.wait()
 try:
  state['status']='running';code=run('scripts/r31_oracle_worker.py','worker');state['worker_exit_code']=code
  if code==0 and not state.get('timed_out',False):state['status']='validating';save('logs/r31_ORACLE_SUPERVISOR.json',state,replace=True);code=run('scripts/r31_oracle_analyze.py','analyzer')
  state.update(status='complete' if code==0 and not state.get('timed_out',False) else 'failed',exit_code=code)
  if state['status']=='complete':state.update(analysis_sha256=sha('results/r31_ORACLE_ANALYSIS.json'),complete_sha256=sha('results/r31_ORACLE_COMPLETE.json'))
 except BaseException as e:
  state.update(status='failed',error=repr(e))
  if process is not None and process.poll() is None:stop(KILL);process.wait()
  raise
 finally:
  elapsed=time.monotonic()-start;state.update(finished_unix=time.time(),CPU_job_wall_seconds=elapsed,overrun=elapsed>180,GPU_worker_seconds=0,inherited_GPU_seconds=total,remaining_GPU_seconds=21600-total)
  if elapsed>180:state['status']='overrun_ineligible'
  assert all(sha(p)==h for p,h in old.items());save('logs/r31_ORACLE_SUPERVISOR.json',state,replace=True)
 if state['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
