import subprocess,tempfile,uuid,time
from pathlib import Path
from unit_history import bootstrap,record_active
from prelaunch_guard import prior_absent

def unit(*args):return subprocess.run(['systemctl','--user',*args],capture_output=True,text=True)
root=unit('show','-p','ControlGroup','--value','--','-.slice').stdout.strip();assert root.startswith('/user.slice/')
with tempfile.TemporaryDirectory() as td:
 name='ouro-abca-history-'+uuid.uuid4().hex+'.service';state=Path(td)/'state'
 try:
  bootstrap(name,state,Path('/sys/fs/cgroup')/root.lstrip('/'))
  p=subprocess.run(['systemd-run','--user','--unit='+name,'--property=Restart=no','--property=KillMode=control-group','/bin/sleep','30'],capture_output=True,text=True);assert p.returncode==0,p.stderr
  x=record_active(name,state);assert not prior_absent(name,x['control_group'])
  unit('stop',name);end=time.monotonic()+10
  while not prior_absent(name,x['control_group']) and time.monotonic()<end:time.sleep(.1)
  assert prior_absent(name,x['control_group'])
  print('SYSTEMD_HISTORY_FIXTURE_PASS unit='+name+' group='+x['control_group'],flush=True)
 finally:unit('stop',name)
