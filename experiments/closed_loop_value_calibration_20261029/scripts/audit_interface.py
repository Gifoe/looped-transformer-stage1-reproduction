"""Check the new head interface cannot use gold or recorded future outcomes."""
from pathlib import Path
import json,hashlib,copy
import numpy as np
from value_core import observation,features,KINDS,ALPHAS,choose
ROOT=Path(__file__).resolve().parents[1]
source=ROOT.parent/'closed_loop_action_value_dev400_20261029/results/development_packed.jsonl'
with source.open(encoding='utf8') as f:rows=[json.loads(l) for l in f]
checked=0
for row in rows:
    altered=copy.deepcopy(row);altered['gold']=-999;altered['actions']={'malicious_future':'must not enter features'};altered['terminal']={}
    for kind in KINDS:
        for action in ALPHAS[1:]:
            a=features(observation(row),kind,action);b=features(observation(altered),kind,action)
            assert (a is None and b is None) or np.array_equal(a,b);checked+=1
    obs=observation(row)
    try:features({**obs,'gold':row['gold']},'input','-1.0')
    except AssertionError:pass
    else:raise AssertionError('Extra fields should fail closed')
empty=dict(decision={},controls={a:dict(margin=0.,entropy=0.) for a in ALPHAS})
assert choose(empty,dict(kind='input',actions={}))==('0.0',[0.,0.,0.,0.])
report=dict(scope='Interface test only; no fitting or new outcome evaluation',real_observations=len(rows),feature_pairs_checked=checked,future_outcome_and_gold_mutations_ignored=True,extra_fields_rejected=True,missing_probes_default_to_adaptive=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),value_core_sha256=hashlib.sha256((ROOT/'scripts/value_core.py').read_bytes()).hexdigest())
(ROOT/'results/INTERFACE_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
