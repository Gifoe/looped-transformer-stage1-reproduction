"""Independent historical endpoint cross-tab; no new data or model selection."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

P=Path(__file__).resolve().parent
raw=(P.parent/'results/r11_test_selected.jsonl').read_bytes()
data=[json.loads(x) for x in raw.decode('utf8').splitlines()]
groups=collections.defaultdict(list)
rescue=collections.Counter()
damage=collections.Counter()
for q in data:
    stable={}
    agreement={}
    for k in (1,4):
        predictions=[]
        for rotation in q['cyclic_rotations']:
            order=rotation['order']
            mapped=np.asarray(rotation['sum_scores'][f'native_k{k}'])
            predictions.append(order[int(mapped[order].argmax())])
        counts=collections.Counter(predictions)
        n=len(predictions)
        stable[k]=len(counts)==1
        agreement[k]=sum(c*(c-1) for c in counts.values())/(n*(n-1))
    c1=q['correct']['cyclic_joint1_only']
    c4=q['correct']['cyclic_joint4_only']
    groups[(stable[1],c1,stable[4],c4)].append(q['id'])
    change='increased' if agreement[4]>agreement[1] else 'decreased' if agreement[4]<agreement[1] else 'unchanged'
    if not c1 and c4:rescue[change]+=1
    if c1 and not c4:damage[change]+=1
assert sum(len(v) for v in groups.values())==1172
assert len(groups[(True,False,True,True)])==47
assert len(groups[(True,True,True,False)])==1
result=dict(status='passed_independent_question_level_endpoint_cross_tab',n=1172,
    input_sha256=hashlib.sha256(raw).hexdigest(),
    cross_table=[dict(K1_unanimous=a,K1_cyclic_correct=b,K4_unanimous=c,K4_cyclic_correct=d,n=len(v)) for (a,b,c,d),v in sorted(groups.items())],
    unanimous_wrong_to_unanimous_correct_ids=groups[(True,False,True,True)],
    unanimous_correct_to_unanimous_wrong_ids=groups[(True,True,True,False)],
    cyclic_rescues_pairwise_agreement_change=dict(rescue),
    cyclic_damages_pairwise_agreement_change=dict(damage),
    interpretation='Endpoints only. Does not assert unchanged intervening K2/K3 or an internal semantic cause. Cyclic rotations are correlated within a question.')
(P/'root_cross_tab.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if not k.endswith('_ids')},indent=2))
