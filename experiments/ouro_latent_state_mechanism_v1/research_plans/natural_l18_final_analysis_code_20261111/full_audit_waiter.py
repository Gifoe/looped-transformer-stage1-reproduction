"""After successful complete collection, launch one independent CUDA audit."""
import os,fcntl,json,subprocess,time,hashlib
from pathlib import Path
ROOT=Path('/home/luansiyu/loop-latent');RUN=ROOT/'research_plans/natural_l18_full_run_20261111';CODE=ROOT/'research_plans/natural_l18_final_analysis_code_20261111/full_audit.py';EXPECTED='db8745e54f681392c2f27a271bed5853482ab7beafb7bf62519beb774275141d';UNIT='ouro-natural-l18-full-20261111.service';AUDIT='ouro-natural-l18-independent-audit-20261111'
def main():
 f=(ROOT/'research_plans/natural_l18_full_audit_wait.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);print('WAIT_L18_COLLECTOR',flush=True)
 while True:
  x=subprocess.check_output(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','Result'],text=True,timeout=15);d=dict(l.split('=',1) for l in x.splitlines())
  if d['ActiveState']=='active':time.sleep(15);continue
  if d['ActiveState']!='inactive' or d['Result']!='success':raise RuntimeError('collector failed '+str(d))
  break
 receipt=RUN/'receipt.json'
 if not receipt.exists():raise RuntimeError('collector receipt missing')
 data=json.loads(receipt.read_text())
 if data.get('status')!='FULL_L18_COLLECTION_PENDING_INDEPENDENT_COVERAGE' or data.get('counts')!={'train':1119,'validation':299,'test':1172}:raise RuntimeError('collector incomplete')
 if hashlib.sha256(CODE.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('audit code drift')
 fd=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(fd,fcntl.LOCK_UN)
 finally:os.close(fd)
 envpath=str(ROOT/'research_plans/natural_l18_probe_code_20261111')+':'+str(ROOT/'research_plans/natural_branch_full_code_v5_20261111')
 cmd=['systemd-run','--user','--unit='+AUDIT,'--property=KillMode=control-group','--property=Restart=no','--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','--setenv=PYTHONPATH='+envpath,str(ROOT/'conda-env/bin/python'),'-u',str(CODE)]
 subprocess.run(cmd,check=True,timeout=30);print('L18_INDEPENDENT_AUDIT_LAUNCHED',flush=True)
if __name__=='__main__':main()
