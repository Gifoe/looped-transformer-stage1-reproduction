"""R21 CPU-only hard20minute supervision; no GPU worker ledger mutation."""
import os,time,subprocess,signal,sys
from r21_common import ROOT,gate,save

def main():
 gate();start=time.monotonic();state=dict(status='starting',GPU_worker_seconds=0,hard_CPU_seconds=1200);p=None
 try:
  with (ROOT/'logs/r21_worker.out').open('x') as out,(ROOT/'logs/r21_worker.err').open('x') as err:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',R21_CPU_DEADLINE_MONOTONIC=str(start+1195))
   p=subprocess.Popen([sys.executable,'-B','-u','scripts/r21_readout_oracle.py'],cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=os.name!='nt');state.update(worker_pid=p.pid,status='running')
   try:code=p.wait(timeout=max(0,start+1199-time.monotonic()))
   except subprocess.TimeoutExpired:
    state['timed_out']=True
    if os.name=='nt':p.kill()
    else:os.killpg(p.pid,signal.SIGKILL)
    code=p.wait()
   state.update(status='complete' if code==0 else 'incomplete',exit_code=code)
 finally:
  state['CPU_wall_seconds']=time.monotonic()-start;state['overrun']=state['CPU_wall_seconds']>1200;save('logs/r21_SUPERVISOR.json',state)
if __name__=='__main__':main()
