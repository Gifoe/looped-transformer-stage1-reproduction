"""Only R2 training readouts and CPU comparison; no implicit test permission."""
import subprocess
from slcd_core import ROOT,save
for split in ['discovery','selection','confirmation']:
    subprocess.run([str(ROOT/'env/bin/python'),'-u','scripts/slcd_supervise.py','--name','r4_'+split,'--seconds','1200','--','slcd_r4_worker.py','--split',split,'--job','development','--recipes','configs/slcd_R4_RECIPES.json'],cwd=ROOT,check=True)
subprocess.run([str(ROOT/'env/bin/python'),'-u','scripts/slcd_r4_evaluate.py','train'],cwd=ROOT,check=True)
save('results/slcd_R4_TRAIN_CHAIN_COMPLETE.json',dict(status='complete',train_only=True))
