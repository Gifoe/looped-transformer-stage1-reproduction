"""CPU-only30min wall supervision,25min stops new state files."""
import os,time,sys,subprocess,signal
from r22_common import ROOT,gate,save

def main():
 gate();start=time.monotonic();state=dict(status='starting',GPU_worker_seconds=0,hard_CPU_seconds=1800);p=None
 try:
  with (ROOT/'logs/r22_worker.out').open('x') as out,(ROOT/'logs/r22_worker.err').open('x') as err:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',R22_STOP_NEW_FILE_MONOTONIC=str(start+1500))
   p=subprocess.Popen([sys.executable,'-B','-u','scripts/r22_features_worker.py'],cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=os.name!='nt');state.update(worker_pid=p.pid,status='running')
   try:code=p.wait(timeout=max(0,start+1799-time.monotonic()))
   except subprocess.TimeoutExpired:
    state['timed_out']=True
    if os.name=='nt':p.kill()
    else:os.killpg(p.pid,signal.SIGKILL)
    code=p.wait()
   state.update(status='complete' if code==0 else 'incomplete',exit_code=code)
 finally:
  state['CPU_wall_seconds']=time.monotonic()-start;state['overrun']=state['CPU_wall_seconds']>1800;save('logs/r22_SUPERVISOR.json',state)
if __name__=='__main__':main()
