"""Independent analyzer certificate counterexamples; synthetic only."""
from pathlib import Path
import json,copy,hashlib,ast
from r31_oracle_math import coefficients,margins,pack
from r31_oracle_solver import propose
from r31_oracle_analyze import certify
ROOT=Path(__file__).resolve().parents[1]
def reject(f):
 try:f()
 except AssertionError:return
 raise AssertionError('Corrupt certificate accepted')
def main():
 pieces=[([-2.,-1.],[-.5,-2.5]),([-1.,-1.5],[-1.,-1.5]),([-5.,-5.],[-5.,-5.]),([-5.,-5.],[-5.,-5.])];chains=[['shared_1','shared_2'] for _ in pieces];problem=coefficients(pieces,chains,[1]*4,[0.]*4);row=dict(id='synthetic',split='synthetic',gold=0,background={},chains=chains,characters=[1]*4,late=[0.]*4,problem=problem);wrong,a,d,b=margins(problem,0);proposal=propose(a,d,b);r={k:row[k] for k in ['id','split','gold','background','chains','characters']};r.update(saved_late=row['late'],keys=problem['keys'],zero_keys=problem['zero_keys'],a=[pack(v) for v in problem['a']],d=[[pack(v) for v in line] for line in problem['d']],B=[pack(v) for v in problem['B']],wrong_candidates=wrong,proposal=proposal)
 assert certify(row,r)=='certified_reachable';bad=copy.deepcopy(r);bad['proposal']['MILP']['status']=2;bad['proposal']['LP']['status']=4;bad['proposal']['raw_MILP_result']['status']=2;assert certify(row,bad)=='certified_reachable'
 bad=copy.deepcopy(r);bad['proposal']['witness_margin_lower']=pack(100);reject(lambda:certify(row,bad))
 bad=copy.deepcopy(r);bad['keys'][0]='candidate_specific_cheat';reject(lambda:certify(row,bad))
 bad=copy.deepcopy(r);bad['proposal']['dual_weights'][0]=pack(2);reject(lambda:certify(row,bad))
 for p in ['scripts/r31_oracle_worker.py','scripts/r31_oracle_analyze.py','scripts/r31_oracle_supervise.py']:
  tree=ast.parse((ROOT/p).read_text(encoding='utf8'));assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any('torch' in a.name or 'transformers' in a.name for a in (n.names if isinstance(n,ast.Import) else [])) for n in ast.walk(tree))
 source=(ROOT/'scripts/r31_oracle_analyze.py').read_text(encoding='utf8');assert 'propose(' not in source and "sup['status']=='complete'" in source and "<=180" in source
 result=dict(status='passed_synthetic_independent_certificate_pipeline',solver_status_not_a_certificate=True,corrupt_lower_key_simplex_rejected=True,clean180s_join_required=True,real24_oracle_run=False,model_loaded=False,GPU_worker_seconds=0,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/r31_oracle_solver.py','scripts/r31_oracle_worker.py','scripts/r31_oracle_analyze.py','scripts/r31_oracle_supervise.py','scripts/r31_oracle_pipeline_checks.py']})
 with (ROOT/'results/r31_ORACLE_CPU_PIPELINE_CHECKS.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
