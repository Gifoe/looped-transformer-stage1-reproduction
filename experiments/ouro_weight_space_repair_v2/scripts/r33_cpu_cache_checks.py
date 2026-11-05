"""Reject incompatible cached ABI via in-memory mutations, never old files."""
import copy,json,hashlib
from pathlib import Path
import r30_cache_abi as legacy
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=legacy.read;checks=[]
 variants=[('results/r28_NATIVE_STATE_INDEX.json','primitive'),('results/r28_NATIVE_STATE_INDEX.json','head_shape'),('results/r28_NATIVE_STATE_INDEX.json','raw_state_shape'),('results/r28_NATIVE_STATE_INDEX.json','weights'),('configs/r28_TRUE_PREFIX_TASK_INVENTORY.json','read_position')]
 for path,mode in variants:
  def altered(p,path=path,mode=mode):
   x=original(p)
   if p!=path:return x
   x=copy.deepcopy(x)
   if mode=='primitive':x['tasks'][0]['primitive']='old_teacherforced_trajectory'
   elif mode=='head_shape':x['tasks'][0]['head_input_shape']=[1,2,2048]
   elif mode=='raw_state_shape':x['tasks'][0]['states'][0]['shape']=[1,2,2048]
   elif mode=='weights':x['tasks'][0]['weights_sha256']='incorrect_weight_identity'
   elif mode=='read_position':x['tasks'][0]['read_position']=-1
   return x
  legacy.read=altered
  try:
   try:legacy.validate()
   except AssertionError:checks.append(mode+'_rejected')
   else:raise AssertionError('Incompatible ABI accepted: '+mode)
  finally:legacy.read=original
 assert legacy.read is original
 result=dict(status='passed_CPU_negative_cache_ABI',checks=checks,old_files_mutated=False,mutation_scope='Only in-memory JSON objects returned by patched reader; original reader restored after every fixture',model_loaded=False,new_head_run=False,GPU_worker_seconds=0,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/r30_cache_abi.py','scripts/r33_cpu_cache_checks.py']})
 with (ROOT/'results/r33_CPU_CACHE_NEGATIVE_CHECKS.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
