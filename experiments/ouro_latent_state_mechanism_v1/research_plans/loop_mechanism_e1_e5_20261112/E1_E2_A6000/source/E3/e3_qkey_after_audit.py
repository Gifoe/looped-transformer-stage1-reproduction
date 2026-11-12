"""Resume E1 and analyze producer64 only after completed independent model audit."""
import json,time,subprocess,os
from pathlib import Path
R=Path('/home/luansiyu/loop-latent');B=R/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';audit=B/'E3/qkey_producer_train64/independent_audit.json'; unit='ouro-e3-qkey64-bare-audit-20261112';py=str(R/'conda-env/bin/python')
print('WAITING_QKEY64_INDEPENDENT_FORWARD_AUDIT',flush=True)
while True:
 p=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','Result'],capture_output=True,text=True,check=True);state=dict(l.split('=',1) for l in p.stdout.splitlines() if '=' in l)
 if state.get('ActiveState') not in ('active','activating'):
  if state.get('Result')!='success' or not audit.exists():raise RuntimeError('Q-key audit failed: no E1 restart or effect analysis')
  a=json.loads(audit.read_text())
  if a.get('status')!='E3_QKEY64_INDEPENDENT_BARE_FORWARD_AND_CUDA_SCORE_AUDIT_PASSED' or a.get('count')!=64:raise RuntimeError('audit receipt invalid')
  break
 time.sleep(15)
paths=[B/'E1',R/'research_plans/natural_qkv_eight_corner_20261112',R/'research_plans/natural_qkv_probe_code_20261112',R/'research_plans/natural_l18_probe_code_20261111',R/'research_plans/natural_branch_full_code_v5_20261111'];envarg='PYTHONPATH='+':'.join(map(str,paths));E=B/'E1';cmd=py+' -u '+str(E/'e1_train_remainder_collector.py')+' && '+py+' -u '+str(E/'e1_train_remainder_audit.py')
subprocess.run(['systemd-run','--user','--unit=ouro-e1-train-resume7-20261112','--property=KillMode=control-group','--setenv='+envarg,'/bin/bash','-c',cmd],check=True);print('E1_RESUMED',flush=True)
os.nice(19);subprocess.run([py,'-u',str(B/'E3/e3_qkey_producer64_analysis.py')],check=True);print('QKEY64_ANALYZED',flush=True)
