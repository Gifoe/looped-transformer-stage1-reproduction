"""Synthetic child only: actual PQR run_owned lifecycle, CPU no GPU."""
import os,sys,subprocess,signal,fcntl,time,pathlib,tempfile,json
P=pathlib.Path('/home/luansiyu/loop-latent');sys.path.insert(0,str(P/'repo/experiments/ouro_latent_state_mechanism_v1'))
from pqr_supervision import run_owned
from ouro_watch import group_live_members

def wait(p):
 deadline=time.monotonic()+20
 while not p.exists():
  if time.monotonic()>deadline:raise RuntimeError('timeout')
  time.sleep(.05)
def controller(d):
 lock=os.open(d/'lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX)
 def handler(s,f):raise InterruptedError('signal')
 signal.signal(signal.SIGTERM,handler)
 code='import signal,subprocess,sys,time,pathlib; signal.signal(signal.SIGTERM,lambda s,f:(time.sleep(2),sys.exit(0))); subprocess.Popen([sys.executable,"-c","import time;time.sleep(120)"]);pathlib.Path(sys.argv[1]).write_text("ready");time.sleep(120)'
 def monitor(proc,elapsed):
  p=d/'pid.json'
  if not p.exists():p.write_text(json.dumps({'pgid':proc.pid}))
 try:run_owned([sys.executable,'-c',code,str(d/'child_ready')],d/'child.log',d,os.environ.copy(),monitor,period=.05)
 except InterruptedError:pass
 finally:(d/'finished').write_text('yes');os.close(lock)
if __name__=='__main__' and len(sys.argv)>1:controller(pathlib.Path(sys.argv[1]))
elif __name__=='__main__':
 with tempfile.TemporaryDirectory(dir=P/'research_plans/pqr_eight_corner_code_20261111') as td:
  d=pathlib.Path(td);p=subprocess.Popen([sys.executable,__file__,str(d)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);wait(d/'child_ready');wait(d/'pid.json');pgid=json.loads((d/'pid.json').read_text())['pgid'];os.kill(p.pid,signal.SIGTERM);time.sleep(.3);fd=os.open(d/'lock',os.O_RDWR)
  try:
   try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:pass
   else:raise AssertionError('lock released before child cleanup')
  finally:os.close(fd)
  try:p.wait(timeout=20)
  finally:
   if p.poll() is None:p.kill();p.wait()
  assert p.returncode==0 and (d/'finished').exists() and not group_live_members(pgid)
  fd=os.open(d/'lock',os.O_RDWR);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);os.close(fd)
  leader='import subprocess,sys,pathlib; subprocess.Popen([sys.executable,"-c","import time;time.sleep(120)"]);pathlib.Path(sys.argv[1]).write_text("ready")'
  ids=[]
  def monitor(proc,elapsed):ids.append(proc.pid)
  run_owned([sys.executable,'-c',leader,str(d/'leader_ready')],d/'leader.log',d,os.environ.copy(),monitor,period=.02)
  assert (d/'leader_ready').exists() and ids and not group_live_members(ids[0])
 print('PQR_RUN_OWNED_SIGTERM_LOCK_GRANDCHILD_CPU_PASS')
