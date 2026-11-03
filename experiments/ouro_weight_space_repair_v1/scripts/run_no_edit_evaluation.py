"""Frozen baseline regression after an explicit discovery-stage no-edit decision.

No directions or methods are generated here. Official test is first downloaded
after FINAL_LOCK. Every scoring worker is serialized and charged to the8h ledger.
"""
import hashlib,json,os,subprocess,sys,time
from runtime import ROOT,save

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 decision_path=ROOT/'configs/STAGE_C_DECISION.json';decision=json.loads(decision_path.read_text())
 assert decision['halt_weight_editing'] is True and decision['primary_weight_edit'] is None
 recipe_path=ROOT/'configs/frozen_baseline_recipe.json';recipe=json.loads(recipe_path.read_text())
 assert recipe['candidate_definitions_frozen'] is True and recipe['weight_edits']==[]
 assert recipe['readout_chord_coefficients']==[.125,.25,.5]
 assert json.loads((ROOT/'results/READINESS.json').read_text())['strict_zero_error_original_parity'] is True
 state=dict(status='running',started_unix=time.time(),steps=[],decision_sha256=digest(decision_path),recipe_sha256=digest(recipe_path))
 def run(name,args,timeout):
  step=dict(name=name,args=args,status='running',started_unix=time.time());state['steps'].append(step);save('logs/FROZEN_EVALUATION_PIPELINE.json',state)
  with (ROOT/'logs'/f'eval_pipeline_{name}.out').open('a') as out,(ROOT/'logs'/f'eval_pipeline_{name}.err').open('a') as err:
   p=subprocess.run([sys.executable,str(ROOT/'scripts'/args[0]),*args[1:]],cwd=ROOT,stdout=out,stderr=err,timeout=timeout)
  step.update(status='complete' if p.returncode==0 else 'failed',exit_code=p.returncode,finished_unix=time.time());save('logs/FROZEN_EVALUATION_PIPELINE.json',state)
  if p.returncode:raise RuntimeError(f'Frozen evaluation stopped at {name}')
 def lock(name,extra):
  path=ROOT/'configs'/name
  required=dict(candidate_definitions_frozen=True,recipe_sha256=digest(recipe_path),decision_sha256=digest(decision_path),primary_weight_edit=None,all_parameters_frozen=True,model_revision=recipe['model_revision'],**extra)
  if path.exists():
   current=json.loads(path.read_text());assert all(current[k]==v for k,v in required.items())
  else:save('configs/'+name,dict(**required,frozen_unix=time.time()))
 def evaluate(split,seconds):
  completion=ROOT/'results'/('frozen_'+split)/'COMPLETE.json'
  if completion.exists():
   c=json.loads(completion.read_text());assert c['recipe_sha256']==digest(recipe_path) and c['n']=={'selection':192,'validation':299,'test':1172}[split];return
  run(split,['supervise.py','--name','frozen_'+split,'--seconds',str(seconds),'--','run_frozen_eval.py','--split',split,'--recipe',recipe_path.name],seconds+60)
 try:
  # No primary edit is fabricated to populate a table. The same previously
  # declared scalar/readout controls are frozen before held-out evaluation.
  lock('CANDIDATE_LOCK.json',dict(official_validation_opened=False,source_of_directions='none',methods=recipe['methods']))
  evaluate('selection',1200)
  run('validation_download',['download_arc_split.py','--split','validation'],100)
  run('validation_manifest',['build_eval_manifests.py','--split','validation'],120)
  evaluate('validation',1600)
  lock('FINAL_LOCK.json',dict(test_evaluation_authorized=True,test_previously_opened=False,test_role='locked_regression_benchmark_previously_used_in_earlier_work',methods=recipe['methods'],test_method_or_depth_selection_forbidden=True,selected_scalar_coefficient=None,primary_contrast='unavailable_no_weight_candidate',secondary_contrasts=['native_k4_vs_native_k5','native_k5_vs_native_k6','native_k4_vs_loopcd_fixed','native_k4_vs_loopcd_adaptive','native_k5_vs_each_prefrozen_chord']))
  run('test_download',['download_arc_split.py','--split','test'],100)
  run('test_manifest',['build_eval_manifests.py','--split','test'],120)
  evaluate('test',6000)
  run('analysis',['analyze_evaluation.py','--include-test'],180)
  state.update(status='complete',finished_unix=time.time());save('logs/FROZEN_EVALUATION_PIPELINE.json',state)
 except BaseException as exc:
  state.update(status='failed',error=repr(exc),finished_unix=time.time());save('logs/FROZEN_EVALUATION_PIPELINE.json',state);raise
if __name__=='__main__':main()
