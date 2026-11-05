"""Once-only complete24 CPU oracle proposal job; no inference/training path."""
import os,time,traceback,platform,sys
from r31_oracle_common import read,sha,save,gate
from r31_oracle_math import pack,margins
from r31_oracle_solver import propose
from r31_oracle_inputs import load
from r31_oracle_versions import versions
def main():
 start=time.monotonic();spec=gate();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';identity=versions();assert identity==spec['solver_identity'];rows,audit=load();records=[];fatal=None
 try:
  for row in rows:
   assert time.time()+8<float(os.environ['R31_CPU_DEADLINE_UNIX']),'Stop before starting a question without bounded solver/IO reserve'
   p=row['problem'];wrong,a,d,b=margins(p,row['gold']);proposal=propose(a,d,b,limit=3.);records.append(dict(id=row['id'],split=row['split'],gold=row['gold'],background=row['background'],chains=row['chains'],characters=row['characters'],saved_late=row['late'],keys=p['keys'],zero_keys=p['zero_keys'],a=[pack(x) for x in p['a']],d=[[pack(v) for v in line] for line in p['d']],B=[pack(x) for x in p['B']],wrong_candidates=wrong,proposal=proposal))
   save('results/r31_ORACLE_PROGRESS.json',dict(completed_questions=len(records),target_questions=24,elapsed_cpu_wall_seconds=time.monotonic()-start),replace=True)
  assert len(records)==24
 except BaseException as e:fatal=dict(error=repr(e),traceback=traceback.format_exc())
 save('results/r31_ORACLE_RAW.json',dict(status='complete' if fatal is None else 'incomplete',records=records,input_audit=audit,protocol_sha256=sha('configs/r31_SHARED_PREFIX_ORACLE_LOCK.json'),fatal=fatal,platform=platform.platform(),python=sys.version,solver_identity=identity,scope='Retrospective capacity oracle only; never a deployed selector or claimed accuracy gain'))
 save('results/r31_ORACLE_COMPLETE.json',dict(status='complete' if fatal is None else 'incomplete',questions=len(records),raw_sha256=sha('results/r31_ORACLE_RAW.json'),protocol_sha256=sha('configs/r31_SHARED_PREFIX_ORACLE_LOCK.json'),elapsed_cpu_wall_seconds=time.monotonic()-start,fatal=fatal,model_loaded=False,GPU_worker_seconds=0,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']))
 if fatal is not None:raise RuntimeError(fatal['error'])
if __name__=='__main__':main()
