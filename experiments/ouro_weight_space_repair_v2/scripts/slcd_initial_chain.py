"""Finite original-state jobs, each charged to the independent SLCD ledger."""
import subprocess,os,json
from slcd_core import ROOT,save


def main():
    for name,arguments,seconds in [('head_discovery',['--split','discovery','--job','initial','--reuse-states'],1800),
        ('collect_selection',['--split','selection','--job','initial'],1800)]:
        result=subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'scripts/slcd_supervise.py'),
            '--name',name,'--seconds',str(seconds),'--','slcd_token_worker.py',*arguments],cwd=ROOT)
        if result.returncode:raise RuntimeError(name+' failed')
    save('results/slcd_INITIAL_CHAIN_COMPLETE.json',dict(status='complete',splits=['discovery','selection']))


if __name__=='__main__':main()
