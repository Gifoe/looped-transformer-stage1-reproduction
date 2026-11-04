"""Low-CPU-priority shared run of exactly one bounded probe and audit."""
import os,subprocess,time
from r15_common import ROOT,save,gate


def status(value,**fields):
    save('logs/r15_LAUNCH_STATUS.json',dict(status=value,pid=os.getpid(),updated_unix=time.time(),**fields))


def main():
    gate();os.nice(15)
    # No other process is stopped or reconfigured. Scientific definitions,
    # cohort and 240s+15s worker limit remain identical to the idle-wait version.
    status('running_bounded_shared_probe',CPU_niceness=os.nice(0))
    subprocess.run([str(ROOT/'env/bin/python'),'-B','-u','scripts/r15_supervise.py'],cwd=ROOT,check=True)
    status('running_frozen_association_analysis')
    subprocess.run([str(ROOT/'env/bin/python'),'-B','-u','scripts/r15_analyze.py'],cwd=ROOT,check=True)
    status('complete_observational_probe',new_method_selected=False,heldout_accessed=False)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed_requires_inspection',error=repr(error));raise
