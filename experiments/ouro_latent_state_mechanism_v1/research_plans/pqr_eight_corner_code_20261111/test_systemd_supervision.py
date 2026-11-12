"""Only a newly created own systemd user unit; no GPU and no existing service touched."""
import os,sys,subprocess,signal,fcntl,time,pathlib,tempfile,json,uuid
P=pathlib.Path('/home/luansiyu/loop-latent');sys.path.insert(0,str(P/'repo/experiments/ouro_latent_state_mechanism_v1'))
from pqr_supervision import run_owned
from ouro_watch import group_live_members

def wait(p,seconds=20):
 deadline=time.monotonic()+seconds
 while not p.exists():
  if time.monotonic()>deadline:raise RuntimeError('timeout')
  time.sleep(.05)
def controller(d):
 lock=os.open(d/'lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX)
 code='import subprocess,sys,time,pathlib;subprocess.Popen([sys.executable,"-c","import time;time.sleep(120)"]);pathlib.Path(sys.argv[1]).write_text("ready");time.sleep(120)'
 def monitor(proc,elapsed):
  p=d/'pid.json'
  if not p.exists():p.write_text(json.dumps({'pgid':proc.pid}))
 try:run_owned([sys.executable,'-c',code,str(d/'child_ready')],d/'child.log',d,os.environ.copy(),monitor,period=.05)
 finally:os.close(lock)
if __name__=='__main__' and len(sys.argv)>1:controller(pathlib.Path(sys.argv[1]))
elif __name__=='__main__':
 with tempfile.TemporaryDirectory(dir=P/'research_plans/pqr_eight_corner_code_20261111') as td:
  d=pathlib.Path(td);unit='pqr-lifecycle-fixture-'+uuid.uuid4().hex[:12];cmd=['systemd-run','--user','--unit='+unit,'--property=KillMode=control-group','--property=Restart=no',sys.executable,__file__,str(d)]
  subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  try:
   wait(d/'child_ready');wait(d/'pid.json');pgid=json.loads((d/'pid.json').read_text())['pgid'];subprocess.run(['systemctl','--user','kill','--kill-who=main','--signal=SIGKILL',unit],check=True)
   deadline=time.monotonic()+25
   while time.monotonic()<deadline:
    if not group_live_members(pgid):break
    time.sleep(.1)
   assert not group_live_members(pgid),'own cgroup descendants survived main SIGKILL'
   fd=os.open(d/'lock',os.O_RDWR)
   try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
   finally:os.close(fd)
  finally:subprocess.run(['systemctl','--user','stop',unit],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 print('PQR_SYSTEMD_CONTROL_GROUP_SIGKILL_CPU_PASS')
