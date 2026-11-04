"""R25 CPU-only synthetic preflight; no real keys/query."""
import os,sys,platform
os.environ['CUDA_VISIBLE_DEVICES']=''
from r25_common import ROOT,sha,save

def main():
 import numpy,torch
 from r25_cpu_checks import main as tests
 out=tests();out.update(source_sha256={p.name:sha(str(p.relative_to(ROOT))) for p in (ROOT/'scripts').glob('r25_*.py')},versions=dict(python=sys.version,numpy=numpy.__version__,torch=torch.__version__,platform=platform.platform()))
 assert not (ROOT/'results/r25_CPU_CHECKS.json').exists();save('results/r25_CPU_CHECKS.json',out);print('R25_CPU_CHECKS_PASS')
if __name__=='__main__':main()
