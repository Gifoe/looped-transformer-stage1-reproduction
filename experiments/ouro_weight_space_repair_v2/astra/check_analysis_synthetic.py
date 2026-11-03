"""CPU-only synthetic check; extracts functions without executing analyzer main."""
import ast
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
source=root/'scripts/analyze_v2.py'
tree=ast.parse(source.read_text(encoding='utf8'))
names=['native_k3','native_k4','native_k5','native_k6','loopcd_k4_fixed','loopcd_k4_adaptive']
namespace=dict(np=np,hashlib=hashlib,METHODS=names)
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'paired','adjust_holm','align','validate'}]
exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
counts=collections.Counter(sum(x) for x in itertools.product([-1,0,1],repeat=3))
multinomial=collections.defaultdict(float)
for rescue in range(4):
    for damage in range(4-rescue):
        same=3-rescue-damage
        multinomial[rescue-damage]+=math.factorial(3)/math.prod(math.factorial(x) for x in [rescue,damage,same])/27
error=max(abs(counts[k]/27-multinomial[k]) for k in counts)
assert error<1e-14

def row(i,correct):
    z=[-1.,-2.] if correct else [-3.,-1.]
    method=dict(sum_scores=[z[0],2*z[1]],normalized_scores=z,prediction=int(not correct),correct=bool(correct),margin=z[0]-z[1])
    return dict(id=str(i),gold=0,prompt_sha256='synthetic',choice_char_lengths=[1,2],choice_token_counts=[1,1],methods={m:dict(method) for m in names})

b=[row(i,c) for i,c in enumerate([1,1,1,0,0,0])]
a=[row(i,c) for i,c in enumerate([0,1,1,1,1,0])]
b,a=namespace['align'](b[::-1],a)
paired=namespace['paired'](b,a,'native_k4','synthetic')
assert paired['corrections']==2 and paired['regressions']==1 and paired['net_correct']==1 and paired['exact_mcnemar_p']==1.
holm=[dict(exact_mcnemar_p=x) for x in [.001,.03,.04,.2,.8,.9]]
namespace['adjust_holm'](holm,'adjusted')
assert np.allclose([x['adjusted'] for x in holm],[.006,.15,.16,.6,1.,1.])
primary=[dict(exact_mcnemar_p=x) for x in [.03,.04]]
namespace['adjust_holm'](primary,'adjusted')
assert np.allclose([x['adjusted'] for x in primary],[.06,.06])
zero=namespace['paired'](b,b,'native_k4','synthetic_zero')
assert zero['exact_mcnemar_p']==1 and zero['paired_bootstrap_95_low_pp']==zero['paired_bootstrap_95_high_pp']==0
bad=[dict(x) for x in a];bad[0]['choice_char_lengths']=[2,2]
try: namespace['align'](b,bad)
except AssertionError: pass
else: raise AssertionError('Mismatched lengths accepted')
result=dict(status='passed',data_scope='synthetic only; no outcome files opened',
    analyzer_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    exhaustive_n3_probability_max_error=error, paired_example=paired,
    holm_six=holm,holm_primary_two=primary,zero_discordance=dict(p=zero['exact_mcnemar_p'],
    low=zero['paired_bootstrap_95_low_pp'],high=zero['paired_bootstrap_95_high_pp']),
    ID_reordering_and_character_normalization_passed=True,mismatched_lengths_rejected=True)
(root/'astra/SYNTHETIC_STATS_CHECK.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(dict(status='passed',analyzer_sha256=result['analyzer_sha256'])))
