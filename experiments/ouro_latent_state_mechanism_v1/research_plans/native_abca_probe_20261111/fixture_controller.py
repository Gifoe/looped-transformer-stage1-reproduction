import os,sys,signal,fcntl
from pathlib import Path
import supervisor_draft as sup
from ouro_watch import install_signal_handlers
G=sup.GIB;good={'gpu':{'free_mib':30*1024},'disk_free_bytes':30*G,'memory':{'MemAvailable':12*G}}
count=0
def sample():
 global count
 count+=1
 if os.environ['FIXTURE_MODE']=='resource' and count>1:return {'gpu':{'free_mib':1},'disk_free_bytes':30*G,'memory':{'MemAvailable':12*G}}
 return good
sup.machine=sample
sup.supervise(Path(__file__).with_name('fixture_worker.py'),os.environ['FIXTURE_LOG'],os.environ.copy(),lockpath=Path(os.environ['FIXTURE_LOCK']))
