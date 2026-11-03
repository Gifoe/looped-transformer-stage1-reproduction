"""Restart accepted science on exact legacy call after archiving old-mask data."""
import json,subprocess,sys,time
from runtime import ROOT,save

def main():
 state=dict(status='running',started_unix=time.time(),steps=[])
 def run(name,args,timeout):
  step=dict(name=name,args=args,started_unix=time.time());state['steps'].append(step);save('logs/ORIGINAL_RESTART_PIPELINE.json',state)
  with (ROOT/'logs'/f'original_restart_{name}.out').open('x') as out,(ROOT/'logs'/f'original_restart_{name}.err').open('x') as err:
   p=subprocess.run([sys.executable,str(ROOT/'scripts'/args[0]),*args[1:]],cwd=ROOT,stdout=out,stderr=err,timeout=timeout)
  step.update(exit_code=p.returncode,finished_unix=time.time());save('logs/ORIGINAL_RESTART_PIPELINE.json',state)
  if p.returncode:raise RuntimeError(f'Original-call restart stopped at {name}')
 try:
  run('readiness',['supervise.py','--name','original_readiness','--seconds','600','--','check_original_call_parity.py','--n','3'],650)
  gate=json.loads((ROOT/'results/original_call_parity/COMPLETE.json').read_text())
  assert all(gate[k] for k in ['exact_original_path_parity','original_repeat_exact','exact_intermediate_depth_parity'])
  save('results/READINESS.json',dict(status='passed',allowed_scoring_path='native_uncached_only',native_call='model.model(input_ids=x,use_cache=False)',strict_zero_error_original_parity=True,real_25shot_train_cases=3,reference_receipt='results/original_call_parity/COMPLETE.json',cached_equivalence=False))
  run('discovery',['supervise.py','--name','original_discovery','--seconds','1800','--','run_discovery.py','--n','192','--parity','0','--uncached','--study','discovery_original'],1850)
  run('diagnostics',['after_original_discovery.py'],4200)
  state.update(status='complete',finished_unix=time.time());save('logs/ORIGINAL_RESTART_PIPELINE.json',state)
 except BaseException as exc:
  state.update(status='failed',error=repr(exc),finished_unix=time.time());save('logs/ORIGINAL_RESTART_PIPELINE.json',state);raise
if __name__=='__main__':main()
