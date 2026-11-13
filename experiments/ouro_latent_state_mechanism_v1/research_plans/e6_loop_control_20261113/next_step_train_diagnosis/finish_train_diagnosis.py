"""Paired offline postprocessing, no fitting or new policy selection."""
import json,collections,gzip,hashlib
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/ouro-e6a');D=R/'next_step_train_diagnosis';P=R/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/e6_loop_control_20261113'
rows=[json.loads(l) for l in (D/'OOF_PREDICTIONS.jsonl').read_text().splitlines()]
def ci(x):
 x=np.array(x);rng=np.random.default_rng(20261113);means=[]
 for _ in range(100):means.extend(x[rng.integers(0,len(x),(100,len(x)))].mean(1))
 return np.quantile(means,[.025,.975]).tolist()
paired=[];conditioned=[]
for setting in ['retrospective','prospective']:
 block={f:{} for f in ['output','combined','unrelated','shuffled']}
 for x in rows:
  if x['setting']==setting:block[x['family']][x['id'],x['depth']]=x
 for base in ['output','unrelated','shuffled']:
  by=collections.defaultdict(list)
  for k,a in block['combined'].items():by[k[0]].append(block[base][k]['brier']-a['brier'])
  vals=[np.mean(by[k]) for k in sorted(by)];paired.append({'setting':setting,'comparison':'combined relative '+base,'n_questions':len(vals),'Brier_improvement':float(np.mean(vals)),'paired_question95':ci(vals)})
 for correct in [True,False]:
  kk=[k for k,a in block['output'].items() if (a['utility_prediction']==a['utility_class'])==correct]
  conditioned.append({'setting':setting,'stratum':'OOF output utility prediction '+('correct' if correct else 'wrong'),'n_transitions':len(kk),'output_Brier':float(np.mean([block['output'][k]['brier'] for k in kk])),'combined_Brier':float(np.mean([block['combined'][k]['brier'] for k in kk])),'scope':'offline descriptive residual stratum, not policy selection'})
(D/'PAIRED_LATENT_VS_CONTROLS.json').write_text(json.dumps(paired,indent=2));(D/'CONDITIONAL_RESIDUAL_STRATA.json').write_text(json.dumps(conditioned,indent=2))
report=D/'TRAIN_DEV_FAILURE_DIAGNOSIS.md';report.write_text(report.read_text().split('\n## Sol interpretation of H0/H1')[0])
with report.open('a') as f:
 f.write('\n## Sol interpretation of H0/H1\n\nThe natural selector oracle leaves48 recoverable K4 errors in1094 Train questions, so lack of headroom alone does not explain failure. Existing geometry gives only a small retrospective proper-loss change; unrelated-layer and shuffled additions also improve the baseline, and prospective combined does not improve. Direct same-question latent-versus-control differences are retained below; no mechanism-specific information gain is established merely because one baseline interval is positive. Retrospective diagnostic action correctness is959 for both output and combined, belowK4=966; prospective combined943 is below output948. This does not validate hidden-state control.\n\n'+json.dumps(paired,indent=2)+'\n\nH2 is NOT RUN. It requires per-layer pre-attention residual states not retained by E6B; current cache contains only layer-averaged x. The next permitted experiment is the single frozen candidate-direction lens family under the director technical and30-GPU-minute budget gates. Do not substitute mean-x, tune old geometry thresholds, or use the current Test for feature selection. E6C remains NO-GO.\n')
out=P/'next_step_train_diagnosis';out.mkdir(exist_ok=True)
for src in sorted(D.iterdir()):
 if not src.is_file():continue
 data=src.read_bytes();dest=out/src.name
 if src.suffix=='.jsonl':dest=dest.with_suffix('.jsonl.gz');data=gzip.compress(data,mtime=0)
 dest.write_bytes(data)
for name in ['train_dev_diagnosis.py','export_train_diagnosis.py','finish_train_diagnosis.py']:(out/name).write_bytes((R/name).read_bytes())
(out/'README.md').write_text('This is a separately frozen post-E6 Train-only diagnostic requested by Astra; all statistics/code/execution are Sol. It does not alter E6 predictions or promote a method. Decompress TRAIN_ONLY_INPUTS.jsonl.gz into a fresh run directory for fitting; no Test/Val data is an allowed fit input. Full OOF records, folds, fixed three-alpha selection, direct paired controls and source/freeze receipts are included. H2 candidate-direction capture and online evaluation: NOT RUN.\n')
print('TRAIN_DIAGNOSIS_POSTPROCESS_COMPLETE',paired,flush=True)
