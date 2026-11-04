"""Wait for two idle observations, then run exactly one bounded probe and audit."""
import os,subprocess,time
from r15_common import ROOT,save,gate


def status(value,**fields):
    save('logs/r15_LAUNCH_STATUS.json',dict(status=value,pid=os.getpid(),updated_unix=time.time(),**fields))


def main():
    gate();start=time.monotonic();idle=0
    while time.monotonic()-start<180:
        result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        utilization=int(result.stdout.splitlines()[0].strip());idle=idle+1 if utilization<=10 else 0
        status('waiting_GPU_idle',observed_utilization=utilization,consecutive_idle_observations=idle)
        if idle>=2:break
        time.sleep(5)
    else:
        status('deferred_busy_GPU_no_worker_started',GPU_worker_seconds=0);return
    status('running_bounded_probe')
    subprocess.run([str(ROOT/'env/bin/python'),'-B','-u','scripts/r15_supervise.py'],cwd=ROOT,check=True)
    status('running_frozen_association_analysis')
    subprocess.run([str(ROOT/'env/bin/python'),'-B','-u','scripts/r15_analyze.py'],cwd=ROOT,check=True)
    status('complete_observational_probe',new_method_selected=False,heldout_accessed=False)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed_requires_inspection',error=repr(error));raise
