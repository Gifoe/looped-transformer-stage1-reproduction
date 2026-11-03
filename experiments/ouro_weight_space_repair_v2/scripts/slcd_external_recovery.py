"""Recover transport-only external failures with the same frozen method/worker."""
import subprocess
from slcd_core import ROOT,read,save,sha
arc=read('results/slcd_ARC_TEST_COMPLETE.json');assert arc['status']=='complete' and arc['n']==1172 and arc['primary_correct']>717
assert sha('scripts/slcd_external_worker.py')=='833593c3cf195d7ff0259c7d9e4bad61d4f2832c6c5446fd4e665ea5baa8868c'
initial='results/slcd_EXTERNAL_INITIAL_ATTEMPT.json'
if not (ROOT/initial).exists():save(initial,read('results/slcd_EXTERNAL_EXECUTION.json'))
tasks={}
for task,seconds in [('hellaswag',14400),('openbookqa',1800)]:
    complete=f'results/slcd_{task}_COMPLETE.json'
    if not (ROOT/complete).exists():
        budget=read('logs/slcd_gpu_budget.json');remaining=budget['limit_seconds']-budget['used_seconds']
        if task=='openbookqa' and remaining<=900:
            tasks[task]=dict(status='not_run',reason='remaining budget below900s',remaining_seconds=remaining);continue
        args=[str(ROOT/'env/bin/python'),'-u','scripts/slcd_supervise.py','--name',task+'_transfer','--seconds',str(min(seconds,int(remaining)-60)),'--','slcd_external_worker.py','--task',task]
        subprocess.run(args,cwd=ROOT,check=True)
    receipt=read(complete);assert receipt['status']=='complete' and receipt['n']=={'hellaswag':10042,'openbookqa':500}[task]
    tasks[task]=dict(status='complete',completion_sha256=sha(complete),recovered_initial_network_failure=True)
save('results/slcd_EXTERNAL_EXECUTION.json',dict(authorized=True,tasks=tasks,initial_attempt_receipt=initial))
subprocess.run([str(ROOT/'env/bin/python'),'-u','scripts/slcd_make_deliverables.py'],cwd=ROOT,check=True)
save('results/slcd_RECOVERY_CHAIN_COMPLETE.json',dict(status='complete',primary=arc['primary'],arc_correct=arc['primary_correct'],external_tasks=tasks))
