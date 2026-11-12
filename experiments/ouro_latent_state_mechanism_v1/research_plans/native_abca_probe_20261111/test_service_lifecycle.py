import os,time,uuid,subprocess,fcntl,sys,json
from pathlib import Path
base=Path(__file__).parent/'fixture_runs';base.mkdir(exist_ok=True)
def live(pid):
 try:
  state=Path('/proc/%d/stat'%pid).read_text().split(') ')[1][0]
  return state!='Z'
 except FileNotFoundError:return False
def locked(p):
 fd=os.open(p,os.O_RDWR)
 try:
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
  except BlockingIOError:return True
 finally:os.close(fd)
def unit(*args):return subprocess.run(['systemctl','--user',*args],capture_output=True,text=True)
for mode in ('resource','term','kill'):
 run=base/(mode+'_'+uuid.uuid4().hex);run.mkdir();name='ouro-abca-fixture-'+uuid.uuid4().hex;pidfile=run/'pids';lock=run/'lock';log=run/'worker.log'
 cmd=['systemd-run','--user','--unit='+name,'--property=KillMode=control-group','--setenv=FIXTURE_MODE='+mode,'--setenv=FIXTURE_PIDS='+str(pidfile),'--setenv=FIXTURE_LOCK='+str(lock),'--setenv=FIXTURE_LOG='+str(log),sys.executable,str(Path(__file__).with_name('fixture_controller.py'))]
 sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])
 try:
  p=subprocess.run(cmd,capture_output=True,text=True)
  if p.returncode:raise RuntimeError('systemd-run '+p.stderr)
  end=time.monotonic()+25
  while not pidfile.exists() and time.monotonic()<end:time.sleep(.1)
  if not pidfile.exists():raise RuntimeError('no pids '+unit('status',name).stdout[-1000:])
  worker,grand=map(int,pidfile.read_text().split());assert locked(lock),'lock not held'
  if mode=='term':
   unit('kill','--kill-who=main','--signal=SIGTERM',name);time.sleep(.3);assert locked(lock),'lock released during TERM cleanup'
  elif mode=='kill':unit('kill','--kill-who=main','--signal=SIGKILL',name)
  end=time.monotonic()+20
  while time.monotonic()<end and (live(worker) or live(grand) or locked(lock)):time.sleep(.2)
  state=unit('show',name,'-p','ActiveState','-p','SubState','-p','Result').stdout.strip()
  assert not live(worker) and not live(grand) and not locked(lock),('cleanup',mode,worker,grand,state)
  assert sentinel.poll() is None,'other process affected'
  print('FIXTURE_PASS',mode,'worker',worker,'grandchild',grand,state.replace('\n',' '),flush=True)
 finally:
  unit('stop',name);sentinel.terminate();sentinel.wait(timeout=5)
