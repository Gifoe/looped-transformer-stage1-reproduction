"""CPU metadata inventory and synthetic tests only; no state payload reads."""
import json,os,sys,platform
os.environ['CUDA_VISIBLE_DEVICES']=''
from r22_common import ROOT,read,save,sha

def main():
 import numpy,scipy,torch
 from r22_cpu_checks import main as tests
 manifest=read('configs/r22_CHALLENGE_MANIFEST.json');entries=[]
 for row in manifest['rows']:
  x=row['state'];p=(ROOT/x['file']).resolve();assert p.is_relative_to(ROOT.resolve()) and p.is_file() and p.stat().st_size==x['bytes'];entries.append(dict(id=row['id'],file=x['file'],bytes=p.stat().st_size,expected_sha256=x['sha256']))
 save('results/r22_CPU_PREFLIGHT.json',dict(status='passed_inventory_and_synthetic_only',n=len(entries),total_cache_bytes=sum(x['bytes'] for x in entries),cache_entries=entries,synthetic=tests(),source_sha256={p:sha('scripts/'+p) for p in ['r22_math.py','r22_statistics.py','r22_cpu_checks.py','r22_cpu_preflight.py']},versions=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=numpy.__version__,scipy=scipy.__version__,torch=torch.__version__),scope='os.stat cache inventory only, no state payload loaded, no real feature/AUC; synthetic values only',GPU_worker_seconds=0))
 print('R22_CPU_PREFLIGHT_PASS',len(entries))
if __name__=='__main__':main()
