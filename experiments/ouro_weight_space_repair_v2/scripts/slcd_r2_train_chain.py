"""Only R2 training readouts and CPU comparison; no implicit test permission."""
import subprocess
from slcd_core import ROOT,save
for split in ['discovery','selection','confirmation']:
    subprocess.run([str(ROOT/'env/bin/python'),'-u','scripts/slcd_supervise.py','--name','r2_'+split,'--seconds','1200','--','slcd_r2_worker.py','--split',split,'--job','development','--recipes','configs/slcd_R2_RECIPES.json'],cwd=ROOT,check=True)
subprocess.run([str(ROOT/'env/bin/python'),'-u','scripts/slcd_r2_evaluate.py','train'],cwd=ROOT,check=True)
save('results/slcd_R2_TRAIN_CHAIN_COMPLETE.json',dict(status='complete',train_only=True))
