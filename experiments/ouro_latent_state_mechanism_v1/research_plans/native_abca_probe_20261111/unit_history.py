"""Disabled bootstrap history primitives; caller serializes all launchers."""
import os,json,subprocess,hashlib
from pathlib import Path
from datetime import datetime,timezone
from persistence import write_once

def fields(unit):
 p=subprocess.run(['systemctl','--user','show',unit,'-p','LoadState','-p','ActiveState','-p','ControlGroup','-p','Restart','-p','Job'],capture_output=True,text=True)
 if p.returncode:raise RuntimeError('systemctl query failed')
 return dict(x.split('=',1) for x in p.stdout.splitlines() if '=' in x)

def bootstrap(unit,state_dir,cgroup_user_root):
 state=Path(state_dir);state.mkdir(exist_ok=True);pending=state/'launch_pending.json';record=state/'previous_unit_controlgroup.json'
 if pending.exists() or record.exists():raise RuntimeError('history exists or previous launch unresolved')
 x=fields(unit)
 if x.get('LoadState')!='not-found' or x.get('ActiveState')!='inactive' or x.get('ControlGroup') or x.get('Job') not in ('','0'):raise RuntimeError('unit not clean for bootstrap')
 root=Path(cgroup_user_root)
 if not root.is_dir():raise RuntimeError('user cgroup root unavailable')
 # Exact unit basename within this account's delegated cgroup only.
 def fail_walk(e):raise RuntimeError('cgroup traversal failed') from e
 for directory,dirs,files in os.walk(root,followlinks=False,onerror=fail_walk):
  if unit in dirs:raise RuntimeError('prior unit cgroup exists')
 pending_obj={'unit':unit,'stage':'BOOTSTRAP_PENDING','systemd_observation':x,'user_cgroup_root':str(root.resolve()),'utc':datetime.now(timezone.utc).isoformat()}
 write_once(pending,(json.dumps(pending_obj,sort_keys=True)+'\n').encode())
 return pending_obj

def record_active(unit,state_dir,cgroup_root=Path('/sys/fs/cgroup')):
 state=Path(state_dir);pending=state/'launch_pending.json';record=state/'previous_unit_controlgroup.json'
 if not pending.is_file() or record.exists():raise RuntimeError('missing pending or existing record')
 old=json.loads(pending.read_text());x=fields(unit);cg=x.get('ControlGroup')
 if old['unit']!=unit or x.get('ActiveState')!='active' or x.get('Restart')!='no' or not cg or Path(cg).name!=unit:raise RuntimeError('unit observation invalid')
 actual=cgroup_root/cg.lstrip('/');actual.resolve().relative_to(cgroup_root.resolve())
 if not actual.is_dir() or actual.is_symlink():raise RuntimeError('observed cgroup missing')
 obj={'unit':unit,'observed_active':True,'control_group':cg,'pending_sha256':hashlib.sha256(pending.read_bytes()).hexdigest(),'systemd_observation':x,'utc':datetime.now(timezone.utc).isoformat()}
 write_once(record,(json.dumps(obj,sort_keys=True)+'\n').encode())
 # Keep pending permanently as an audit marker. A separate committed record certifies it.
 return obj
