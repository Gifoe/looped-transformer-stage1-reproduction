import os,time,uuid,subprocess,fcntl,sys
from pathlib import Path
from prelaunch_guard import prior_absent
base=Path(__file__).parent/'fixture_runs';run=base/('race_'+uuid.uuid4().hex);run.mkdir(parents=True);name='ouro-abca-race-'+uuid.uuid4().hex+'.service';pidfile=run/'pids';lock=run/'lock';log=run/'worker.log'
def unit(*args):return subprocess.run(['systemctl','--user',*args],capture_output=True,text=True)
cmd=['systemd-run','--user','--unit='+name,'--property=KillMode=control-group','--setenv=FIXTURE_MODE=term','--setenv=FIXTURE_PIDS='+str(pidfile),'--setenv=FIXTURE_LOCK='+str(lock),'--setenv=FIXTURE_LOG='+str(log),sys.executable,str(Path(__file__).with_name('fixture_controller.py'))]
try:
 p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==0,p.stderr
 end=time.monotonic()+20
 while not pidfile.exists() and time.monotonic()<end:time.sleep(.05)
 assert pidfile.exists()
 cg=unit('show',name,'-p','ControlGroup','--value').stdout.strip();assert cg.endswith('/'+name),cg
 assert not prior_absent(name,cg),'active old unit accepted'
 unit('kill','--kill-who=main','--signal=SIGKILL',name)
 attempts=acquired=refused_while_populated=0;end=time.monotonic()+8
 while time.monotonic()<end:
  attempts+=1;fd=os.open(lock,os.O_RDWR)
  try:
   try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:time.sleep(.005);continue
   acquired+=1;events=Path('/sys/fs/cgroup'+cg+'/cgroup.events')
   try:pop=dict(line.split() for line in events.read_text().splitlines()).get('populated')
   except FileNotFoundError:pop='0'
   allowed=prior_absent(name,cg)
   if pop=='1':
    if allowed:raise AssertionError('admission during populated old cgroup')
    refused_while_populated+=1
   if allowed and pop=='0':break
  finally:os.close(fd)
  time.sleep(.01)
 if not acquired or not allowed:raise AssertionError('no eventual admission')
 print('RACE_PASS attempts=%d acquired=%d observed_populated_refusals=%d last_populated=%s'%(attempts,acquired,refused_while_populated,pop),flush=True)
finally:unit('stop',name)
