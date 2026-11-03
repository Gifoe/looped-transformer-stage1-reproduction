"""Bounded cached-state neighborhood/reference readouts; no new transformer pass."""
import subprocess
from slcd_core import ROOT,save


def main():
    for split in ['discovery','selection']:
        r=subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'scripts/slcd_supervise.py'),
            '--name','neighbors_'+split,'--seconds','900','--','slcd_token_worker.py','--split',split,
            '--job','neighbors','--reuse-states','--recipes','configs/slcd_TOKEN_NEIGHBOR.json'],cwd=ROOT)
        if r.returncode:raise RuntimeError(split+' neighbor job failed')
    save('results/slcd_NEIGHBOR_CHAIN_COMPLETE.json',dict(status='complete'))


if __name__=='__main__':main()
