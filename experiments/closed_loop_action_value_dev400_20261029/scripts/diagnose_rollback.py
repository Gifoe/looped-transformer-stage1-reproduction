"""Post-primary exploratory ranking comparison; no policy changes or fitting."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
with (ROOT/'results/development_packed.jsonl').open(encoding='utf8') as f:rows=[json.loads(l) for l in f]
rows=[r for r in rows if 'losses' in r['decision']]
values=[];controls=[];delta=[]
for r in rows:
    l=r['decision']['losses'];values.append(min(l['0.0']['A']-l['-1.0']['A'],l['0.0']['B']-l['-1.0']['B']))
    controls.append(r['controls']['-1.0']['margin']-r['controls']['0.0']['margin'])
    delta.append(int(r['actions']['-1.0']['prediction']==r['gold'])-int(r['actions']['0.0']['prediction']==r['gold']))
v=np.array(values);c=np.array(controls);d=np.array(delta);pos=np.flatnonzero(d>0);neg=np.flatnonzero(d<0);ix=np.concatenate([pos,neg])
assert len(pos) and len(neg)
rng=np.random.default_rng(2021);diff=[]
for _ in range(5000):
    b=np.concatenate([rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]);diff.append(roc_auc_score(d[b]>0,v[b])-roc_auc_score(d[b]>0,c[b]))
report=dict(scope='Post-primary exploratory rollback audit; no threshold/sign changes or policy fitting',n_available=len(rows),beneficial=len(pos),harmful=len(neg),proxy_auc=float(roc_auc_score(d[ix]>0,v[ix])),margin_change_auc=float(roc_auc_score(d[ix]>0,c[ix])),paired_auc_difference_ci95=np.quantile(diff,[.025,.975]).tolist(),proxy_positive_count=int((v>0).sum()),proxy_max=float(v.max()),proxy_quantiles=np.quantile(v,[0,.25,.5,.75,1]).tolist(),beneficial_median=float(np.median(v[pos])),harmful_median=float(np.median(v[neg])),comparison_adjustment='Unadjusted exploratory interval, not conditional independence or a confirmation test')
(ROOT/'results/ROLLBACK_DIAGNOSTIC.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
