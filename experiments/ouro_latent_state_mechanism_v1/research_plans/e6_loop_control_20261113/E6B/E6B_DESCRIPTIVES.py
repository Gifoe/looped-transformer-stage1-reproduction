"""Observable feature export and all-four-group descriptive statistics; no controller fit."""
import argparse,json,hashlib,collections
from pathlib import Path
import numpy as np
from E6B_ANALYSIS_v2 import load
from E6B_TRAINING_FREE_CONTROLLER import output_features

def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',nargs='+',required=True);p.add_argument('--labels',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);labels={(r['split'],r['id']):r for r in map(json.loads,Path(a.labels).read_text(encoding='utf-8-sig').splitlines())};splits,sources,core=load(a.capture,labels);feature_rows=[];groups=collections.defaultdict(list);corr=collections.defaultdict(list);names={0:'stable_wrong',1:'rescue',2:'damage',3:'stable_correct'}
 for split,rs in splits.items():
  for r in rs:
   feature_rows.append({'id':r['id'],'split':split,'latent':r['latent'],'unrelated':r['unrelated'],'output':[output_features(r['scores'],d).tolist() for d in range(1,5)],'native_ties':r['ties'],'native_scores':r['scores'],'native_predictions':r['pred'],'replica_baselines':r['replica_baselines'],'available_after_depth':[1,2,3,4]})
   for d in [2,3,4]:
    before=r['pred'][d-2]==r['gold'];after=r['pred'][d-1]==r['gold'];g=3 if before and after else 2 if before else 1 if after else 0
    for fam in ['latent','unrelated','output']:
     v=output_features(r['scores'],d) if fam=='output' else np.asarray(r[fam][d-1]);groups[(split,d,fam,names[g])].append(v);corr[(split,d,fam)].append((v,1 if g==1 else -1 if g==2 else 0))
 summary=[]
 for (split,depth,family,group),vals in groups.items():
  x=np.asarray(vals);summary.append({'split':split,'depth':depth,'family':family,'group':group,'n':len(x),'mean':x.mean(0).tolist(),'std':x.std(0).tolist(),'median':np.median(x,axis=0).tolist()})
 correlations=[]
 for (split,depth,family),vals in corr.items():
  x=np.asarray([v[0] for v in vals]);y=np.asarray([v[1] for v in vals]);ys=y.std();xc=x-x.mean(0);yc=y-y.mean();den=np.sqrt((xc*xc).sum(0)*(yc*yc).sum());r=np.divide((xc*yc[:,None]).sum(0),den,out=np.zeros(x.shape[1]),where=den>0);correlations.append({'split':split,'depth':depth,'family':family,'signed_utility_pearson':r.tolist(),'defined':(den>0).tolist(),'definition':'rescue +1, damage -1, stable groups0; descriptive association, not online gold features or causal utility'})
 (out/'E6B_FEATURES.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in feature_rows),encoding='utf8');(out/'E6B_GROUP_DESCRIPTIVES.json').write_text(json.dumps({'groups':summary,'correlations':correlations,'feature_names':core['features'],'scope':'All four natural groups retained; Gold only defines offline grouping. No method selection or additional confirmatory significance test.'},indent=2));print('FEATURE_EXPORT_COMPLETE',len(feature_rows))
if __name__=='__main__':main()
