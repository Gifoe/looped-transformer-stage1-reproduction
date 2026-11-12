"""Draft old-unit/cgroup prelaunch veto; caller must retain global flock."""
from pathlib import Path
import subprocess

def prior_absent(unit,expected_cgroup,cgroup_root=Path('/sys/fs/cgroup')):
 if not unit.startswith('ouro-abca-') or not unit.endswith('.service') or '/' in unit:raise ValueError('unit identity')
 target=(cgroup_root/expected_cgroup.lstrip('/')).resolve();target.relative_to(cgroup_root.resolve())
 if target.name!=unit:raise ValueError('cgroup identity')
 r=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','ControlGroup','-p','Restart','-p','Job'],capture_output=True,text=True)
 if r.returncode:raise RuntimeError('systemctl unavailable')
 fields=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
 if fields.get('Restart')!='no' or fields.get('Job') not in ('','0'):return False
 if fields.get('ControlGroup') and fields['ControlGroup']!=expected_cgroup:raise RuntimeError('unexpected cgroup')
 if fields.get('ActiveState') not in ('inactive','failed'):return False
 try:
  entries=[line.split() for line in (target/'cgroup.events').read_text().splitlines()]
  if any(len(v)!=2 for v in entries) or len({v[0] for v in entries})!=len(entries):return False
  values=dict(entries)
  if values.get('populated')!='0':return False
 except FileNotFoundError:
  if target.exists():return False
 return True
