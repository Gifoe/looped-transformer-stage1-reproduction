"""Prepare review manifest and synthetic evidence only, never real LP inputs."""
import json,hashlib,ast,platform
from pathlib import Path
import scipy,numpy
from r21_cpu_checks import main
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
if __name__=='__main__':
 fs=sorted((R/'scripts').glob('r21_*.py'));sources={p.name:sha(str(p.relative_to(R))) for p in fs}
 for p in fs:ast.parse(p.read_text(encoding='utf-8-sig'))
 receipt=main();receipt.update(source_sha256=sources,scipy_version=scipy.__version__,python_version=platform.python_version(),numpy_version=numpy.__version__,platform=platform.platform(),python_executable=__import__('sys').executable)
 (R/'results/r21_CPU_CHECKS.json').write_text(json.dumps(receipt,indent=2)+'\n')
 inputs=['results/slcd_r2_discovery_development.jsonl','results/slcd_r2_selection_development.jsonl','results/r7_discovery_development.jsonl','results/r7_selection_development.jsonl','data/slcd_discovery.jsonl','data/slcd_selection.jsonl','r21_ASTRA_DESIGN.md','r21_ROOT_SCOPE_REVIEW.md','results/r21_CPU_CHECKS.json','scripts/slcd_r2_worker.py','scripts/r7_worker.py','scripts/r7_inference.py','scripts/runtime.py']
 inputs += ['results/'+prefix+split+'_development_COMPLETE.json' for prefix in ['slcd_r2_','r7_'] for split in ['discovery','selection']]
 spec=dict(status='DRAFT_PENDING_ROOT_ASTRA_REVIEW',source_sha256=sources,input_sha256={p:sha(p) for p in inputs},cohort='Full discovery384 and selection192 manifest order; no subset/depth search',sets={'H6':'native1..6 normalized scores','H7':'H6 plus original R6 separate upperbound only'},tau=1e-8,numeric_certificate='lambda=-ineqlin.marginals. Clip negatives, positive-denominator simplex normalization, raw negative residual retained. L=min(Aw), U=max(A.Tlambda), delta=64eps64*(d+m+2)*max(1,maxabsA). Bounds L-delta,U+delta cover FP normalization/evaluation. Max raw residual or abs(U-L)>tau ->uncertain; nonpositive denominator raises. Solver success not proof.',scaling='per depth subtract first candidate, shared divisor=max(1,max_depth(maxscore-minscore))',categories=['discrete_reachable','only_convex_reachable','hull_unreachable','numerically_undetermined'],screen='R6wrong H6 strict>=20%, each split>=10%, U<=1% all576; diagnostic only',CPU_wall_limit_seconds=1200,GPU_worker_seconds=0,existing_GPU_ledger_total=18472.29435229581,heldout_access_authorized=False,oracle_gold_weights_not_deployable=True,solver='scipy.linprog highs fixed defaults; solver exception stops incomplete; no tolerance/presolve search')
 (R/'configs/r21_REVIEW_DRAFT.json').write_text(json.dumps(spec,indent=2)+'\n')
 (R/'logs/r21_IMPLEMENTATION_SCOPE.txt').write_text('All new CPU-only source; old files unchanged; no real oracle run. H7 separate upperbound. Native strict consistency asserts beyond tau+delta, tiny positive near-boundary not mislabeled implementation failure.\n')
 print('DRAFT',sha('configs/r21_REVIEW_DRAFT.json'));print('CPU',sha('results/r21_CPU_CHECKS.json'))
