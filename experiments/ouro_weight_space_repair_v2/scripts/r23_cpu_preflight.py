"""R23 synthetic CPU evidence only, no models/data/targets."""
import os,json,sys,platform
os.environ['CUDA_VISIBLE_DEVICES']=''
from r23_common import ROOT,sha,save

def main():
 import torch,numpy
 from r23_cpu_checks import main as tests
 result=tests();result.update(source_sha256={p.name:sha(str(p.relative_to(ROOT))) for p in (ROOT/'scripts').glob('r23_*.py')},versions=dict(python=sys.version,numpy=numpy.__version__,torch=torch.__version__,platform=platform.platform()),scope='Synthetic CPU only; no model/load/real features/GPU/heldout')
 assert not (ROOT/'results/r23_CPU_CHECKS.json').exists();save('results/r23_CPU_CHECKS.json',result);print('R23_CPU_CHECKS_PASS')
if __name__=='__main__':main()
