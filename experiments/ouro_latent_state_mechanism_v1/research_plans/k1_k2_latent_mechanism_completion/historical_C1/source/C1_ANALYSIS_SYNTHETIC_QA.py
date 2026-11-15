"""Hand-computable causal-table/sign and bootstrap checks, no model or gold data."""
import hashlib
import json
from pathlib import Path
from fractions import Fraction
import C1_CAUSAL_ANALYSIS as a

values={'U':1.,'UX':3.,'UA':4.,'N':8.,'UXA':8.,'P':0.,'P_plus_L':2.,
        'U_upstream_norm':5.,'U_upstream_unrelated':6.,
        'U_X_norm':2.,'U_X_unrelated':1.5,'U_A_norm':2.5,'U_A_unrelated':2.,
        'N_X_norm_rollback':7.,'N_X_unrelated_rollback':6.5,
        'N_A_norm_rollback':7.5,'N_A_unrelated_rollback':7.}
expected={'total_upstream_loss':7.,'X_restore':2.,'X_loss':4.,'A_restore':3.,'A_loss':5.,
          'interaction':2.,'X_restore_vs_norm':1.,'X_restore_vs_unrelated':1.5,
          'A_restore_vs_norm':1.5,'A_restore_vs_unrelated':2.,
          'X_loss_vs_norm':3.,'X_loss_vs_unrelated':2.5,
          'A_loss_vs_norm':4.5,'A_loss_vs_unrelated':4.,
          'upstream_loss_vs_norm':4.,'upstream_loss_vs_unrelated':5.,'P_forward_restoration':2.}
actual={k:sum(w*values[n] for n,w in c.items()) for k,c in a.CONTRASTS.items()}
assert actual==expected
assert len(a.ROUTE_TESTS)==len(set(a.ROUTE_TESTS))==12
v,ties=a.score_values([1.,3.,3.,-1.],2,1)
assert ties==[1,2] and v['correct']==0 and v['dynamic_wrong']==0
assert v['meanwrong']==2 and v['fixed_K1_wrong']==0
b=a.interval([[2.,-3.,0.]]*7)
for field in b:assert list(b[field])==[2.,-3.,0.]
root=Path(__file__).parent
out={'status':'PASS','scope':'Hand-computable four-corner/control signs, first-displayed exact ties, exact gaps, constant question-bootstrap. No real-data or model validation.',
     'tested_contrasts':actual,'route_family_n':12,
     'source_sha256':a.sha(a.__file__),'QA_source_sha256':a.sha(__file__)}
(root/'C1_ANALYSIS_SYNTHETIC_QA.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
