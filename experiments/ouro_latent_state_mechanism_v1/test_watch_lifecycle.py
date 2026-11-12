"""Exercise controller SIGTERM and orphaned grandchild cleanup; no GPU."""
import fcntl,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
import ouro_watch as w

def wait_file(p,seconds=15):
 deadline=time.monotonic()+seconds
 while not p.exists():
  if time.monotonic()>deadline:raise RuntimeError('fixture timed out')
  time.sleep(.05)
def controller(root):
 lock=os.open(root/'controller.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX);w.install_signal_handlers()
 script='import subprocess,sys,time,signal,pathlib; signal.signal(signal.SIGTERM,lambda s,f:(time.sleep(2),sys.exit(0))); subprocess.Popen([sys.executable,"-c","import time; time.sleep(120)"]); pathlib.Path(sys.argv[1]).write_text("ready"); time.sleep(120)'
 proc=subprocess.Popen([sys.executable,'-c',script,str(root/'child_ready')],start_new_session=True,preexec_fn=w._child_death_signal)
 wait_file(root/'child_ready');(root/'ready.json').write_text(json.dumps({'pgid':proc.pid}))
 try:
  while True:time.sleep(1)
 finally:
  w.terminate_process_group(proc);(root/'cleaned').write_text('done');os.close(lock)
def race_controller(root):
 lock=os.open(root/'race.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX);w.install_signal_handlers();real=subprocess.Popen
 def racing(*args,**kwargs):
  proc=real(*args,**kwargs);w.subprocess.Popen=real
  (root/'race_child.json').write_text(json.dumps({'pgid':proc.pid}));os.kill(os.getpid(),signal.SIGTERM)
  return proc
 w.subprocess.Popen=racing
 try:
  with w.owned_process([sys.executable,'-c','import time; time.sleep(120)']):pass
 except w.WatchInterrupted:pass
 finally:
  w.subprocess.Popen=real;(root/'race_cleaned').write_text('done');os.close(lock)
def main():
 with tempfile.TemporaryDirectory(prefix='watch_lifecycle_',dir=w.ROOT) as d:
  root=Path(d)
  # Leader exits first; an inherited-group grandchild must still be stopped.
  script='import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(120)"])'
  leader=subprocess.Popen([sys.executable,'-c',script],start_new_session=True,preexec_fn=w._child_death_signal);leader.wait(timeout=10);time.sleep(.2)
  if not w.group_live_members(leader.pid):raise AssertionError('orphan fixture not running')
  w.terminate_process_group(leader);assert not w.group_live_members(leader.pid)
  proc=subprocess.Popen([sys.executable,__file__,'controller',str(root)],stderr=subprocess.DEVNULL);wait_file(root/'ready.json');pgid=json.loads((root/'ready.json').read_text())['pgid'];os.kill(proc.pid,signal.SIGTERM);time.sleep(.4)
  check=os.open(root/'controller.lock',os.O_RDWR)
  try:
   try:fcntl.flock(check,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:pass
   else:raise AssertionError('controller released lock while child still cleaning up')
  finally:os.close(check)
  try:proc.wait(timeout=20)
  finally:
   if proc.poll() is None:proc.kill();proc.wait()
  assert (root/'cleaned').exists() and not w.group_live_members(pgid)
  lock=os.open(root/'controller.lock',os.O_RDWR);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.close(lock)
  race=subprocess.Popen([sys.executable,__file__,'race',str(root)],stderr=subprocess.DEVNULL);race.wait(timeout=20);assert race.returncode==0
  racepgid=json.loads((root/'race_child.json').read_text())['pgid'];assert (root/'race_cleaned').exists() and not w.group_live_members(racepgid)
  lock=os.open(root/'race.lock',os.O_RDWR);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.close(lock)
 print('PASS launch-race SIGTERM, controller SIGTERM, orphan cleanup and lock-held cleanup')
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='controller':controller(Path(sys.argv[2]))
 elif len(sys.argv)>1 and sys.argv[1]=='race':race_controller(Path(sys.argv[2]))
 else:main()
