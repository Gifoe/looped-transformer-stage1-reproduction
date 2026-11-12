"""Disabled launch design. Never execute until lifecycle audit and approval."""
import os,fcntl,subprocess,json,hashlib,time
from pathlib import Path
from prelaunch_guard import prior_absent
from unit_history import bootstrap,record_active
ROOT=Path('/home/luansiyu/loop-latent');UNIT='ouro-abca-prefix-native.service'
def start_once():
 state=ROOT/'research_plans/native_abca_probe_20261111/launch_state'
 # This launch serialization lock is distinct from the global GPU flock.
 fd=os.open(ROOT/'abca_launcher.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  record=state/'previous_unit_controlgroup.json';pending=state/'launch_pending.json';initial=False
  if record.exists():
   if not pending.is_file():raise RuntimeError('missing pending provenance')
   j=json.loads(record.read_text())
   if j.get('unit')!=UNIT or j.get('observed_active') is not True or j.get('pending_sha256')!=hashlib.sha256(pending.read_bytes()).hexdigest():raise RuntimeError('invalid prior cgroup record')
   if not prior_absent(UNIT,j['control_group']):raise RuntimeError('previous unit/cgroup remains')
  else:
   x=subprocess.run(['systemctl','--user','show','-p','ControlGroup','--value','--','-.slice'],capture_output=True,text=True)
   if x.returncode or not x.stdout.strip().startswith('/user.slice/'):raise RuntimeError('user manager cgroup unavailable')
   bootstrap(UNIT,state,Path('/sys/fs/cgroup')/x.stdout.strip().lstrip('/'));initial=True
  cmd=['systemd-run','--user','--unit='+UNIT,'--property=KillMode=control-group','--property=Restart=no',str(ROOT/'conda-env/bin/python'),'-u',str(Path(__file__).with_name('supervisor_draft.py'))]
  p=subprocess.run(cmd,capture_output=True,text=True)
  if p.returncode:raise RuntimeError('systemd start failed: '+p.stderr)
  if initial:
   # Any interruption before this durable record leaves pending and refuses retry.
   record_active(UNIT,state)
  return p.stdout
 finally:os.close(fd)
if __name__=='__main__':raise SystemExit('DISABLED: not reviewed or authorized for GPU')
