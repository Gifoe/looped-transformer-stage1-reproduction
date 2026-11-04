"""One low-priority shared bounded worker, then CPU analysis of complete output."""
import os,subprocess,time
from r19_common import ROOT,gate,save


def main():
    gate();os.nice(15)
    save('logs/r19_LAUNCH_STATUS.json',dict(status='running_bounded_pilot',pid=os.getpid(),updated_unix=time.time()))
    subprocess.run([str(ROOT/'env/bin/python'),'-B','-u','scripts/r19_supervise.py'],cwd=ROOT,check=True)
    subprocess.run([str(ROOT/'env/bin/python'),'-B','-u','scripts/r19_analyze.py'],cwd=ROOT,check=True)
    save('logs/r19_LAUNCH_STATUS.json',dict(status='complete',pid=os.getpid(),updated_unix=time.time(),external_benchmarks_started=False))


if __name__=='__main__':
    try:main()
    except Exception as e:
        save('logs/r19_LAUNCH_STATUS.json',dict(status='failed_requires_inspection',pid=os.getpid(),updated_unix=time.time(),error=repr(e)));raise
