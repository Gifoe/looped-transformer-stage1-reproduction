"""Task-local repair for a moved environment's stale inherited .pth path."""
import os,sys,site
from pathlib import Path
if os.name=='nt':
 from windows_task_affinity import set_self
 set_self('0xffff0000')
inherited=Path(__file__).resolve().parents[2]/'looped_stage1_trajectory_audit_seed2021_v1/.venv/Lib/site-packages'
if inherited.exists():
 # Original .pth was processed after the task venv and before system packages.
 own=str(Path(sys.prefix)/'Lib/site-packages')
 index=next((i for i,p in enumerate(sys.path) if Path(p)==Path(own)),0)
 sys.path.insert(index+1,str(inherited))
