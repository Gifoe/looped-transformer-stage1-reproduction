"""Independent Fraction aggregates plus fresh 18-family cluster-bootstrap replay. No neural execution."""
import pathlib,json,gzip,hashlib,sys
from fractions import Fraction as F
R=pathlib.Path(__file__).resolve().parent.parent
def archive(name):
 p=R/name
 if p.exists():b=p.read_bytes()
 else:
  spec=next(z for z in json.loads((R/'SHARDED_ARTIFACTS.json').read_text())['artifacts'] if z['path']==name);b=b''.join((R/z).read_bytes() for z in spec['parts']);assert hashlib.sha256(b).hexdigest()==spec['sha256'] and len(b)==spec['bytes']
 return [json.loads(l) for l in gzip.decompress(b).decode().splitlines()]
def loadrows(phase):
 if (R/'SHARDED_ARTIFACTS.json').exists():return archive(phase.upper()+'_ROWS.jsonl.gz')
 rows=[]
 for p in sorted((R/('results' if phase=='development' else 'confirmation_results')).glob('SR*.json')):rows+=json.loads(p.read_text())['rows']
 if phase=='development':
  d=R.parent/'ouro-binding-interface-sol-20261117'
  for p in sorted((d/'results').glob('SR*_L11.json')):
   for u in json.loads(p.read_text())['rows']:rows.append({**u,'mask':7})
 return rows
def Y(u,k):
 x=list(map(F,u[k]['scores']['2']['candidate_logits']));return x[u['opposite_index']]-(x[u['decoy_indices'][0]]+x[u['decoy_indices'][1]])/2
def G(u,k):
 x=list(map(F,u[k]['scores']['2']['candidate_logits']));return x[u['opposite_index']]-x[u['correct_index']]
def mean(a):return sum(a,F(0))/len(a)
def dose(rr,sem='semantic',ran='random'):
 t=[];a=[];m=[]
 for u in rr:
  d=u[ran]['audit']['events'][0]['dose'];assert d['target']==u[sem]['audit']['events'][0]['actual_delta_norms'];t+=d['target'];a+=d['actual'];m+=d['matched']
 check=[abs(a-t)<=max(1e-9,.01*t) for a,t in zip(a,t)];assert check==m;total=sum(t);er=sum(abs(a-t) for a,t in zip(a,t))/total if total else 0.;mass=sum(t for t,m in zip(t,m) if m)/total if total else 1.;return er,mass,sum(m),len(t)
def dev():
 rows=loadrows('development');assert len(rows)==3584 and len({(u['id'],u['mask']) for u in rows})==3584;s=json.loads((R/'REGION_SELECTION.json').read_text());profiles={};full=[u for u in rows if u['mask']==7];e7={(f,e):mean([Y(u,'semantic')-Y(u,'random') for u in full if (u['firsthop_factor'],u['endpoint_factor'])==(f,e)]) for f in [0,1] for e in [0,1]}
 for mask in range(1,8):
  rr=[u for u in rows if u['mask']==mask];assert len(rr)==512;p=s['profiles'][str(mask)];four={(f,e):mean([Y(u,'semantic')-Y(u,'random') for u in rr if (u['firsthop_factor'],u['endpoint_factor'])==(f,e)]) for f in [0,1] for e in [0,1]};fine={(f,e,q):mean([Y(u,'semantic')-Y(u,'random') for u in rr if (u['firsthop_factor'],u['endpoint_factor'],u['selector'])==(f,e,q)]) for f in [0,1] for e in [0,1] for q in [0,1]};assert {f'F{f}_E{e}':float(v) for (f,e),v in four.items()}==p['four_strata'];assert {f'F{f}_E{e}_Q{q}':float(v) for (f,e,q),v in fine.items()}==p['eight_selector_points'];ratios={k:float(v/e7[k]) for k,v in four.items()};assert {f'F{f}_E{e}':v for (f,e),v in ratios.items()}==p['retention_ratios'];er,mass,matched,n=dose(rr);qc=er<=.01 and mass>=.99;assert p['dose_QC']['pass']==qc and abs(p['dose_QC']['error_ratio']-er)<1e-12 and abs(p['dose_QC']['matched_target_mass']-mass)<1e-12 and p['dose_QC']['matched']==matched and p['dose_QC']['rows']==n;ok=all(v>F(1,10) and v>=F(4,5)*e7[k] for k,v in four.items()) and all(v>0 for v in fine.values()) and qc;assert ok==p['eligible'];profiles[mask]=(mask.bit_count(),-min(ratios.values()),mask) if ok else None
 eligible=[v for v in profiles.values() if v is not None];sel=min(eligible)[2] if eligible else None;assert s['selected_mask']==sel;print('REGION_REPLAY_DEV_PASS all7profiles/3584rows/Fraction selector',sel);return {'status':'PASS','selected_mask':sel,'selection_sha256':hashlib.sha256((R/'REGION_SELECTION.json').read_bytes()).hexdigest(),'method':'Independent Fraction graphstratum means and minimalregion selection, directdoseQC; no neural replication'}
def confirm():
 import numpy as np
 rows=loadrows('confirmation');assert len(rows)==2048 and len({u['id'] for u in rows})==2048;s=json.loads((R/'REGION_CONFIRMATION.json').read_text());group=[]
 names=[f'E_selected_F{f}_E{e}' for f in [0,1] for e in [0,1]]+[f'retention_F{f}_E{e}' for f in [0,1] for e in [0,1]]+[f'E_full_F{f}_E{e}' for f in [0,1] for e in [0,1]]+['Y_selected_sem_vs_native','G_selected_sem_vs_native','native_K2_accuracy','native_K2_minus_K1_accuracy','E_selected_label0','E_selected_label1']
 for pid in sorted({u['pair_id'] for u in rows}):
  rr=[u for u in rows if u['pair_id']==pid];assert len(rr)==32;d={}
  for f in [0,1]:
   for e in [0,1]:
    r=[u for u in rr if (u['firsthop_factor'],u['endpoint_factor'])==(f,e)];a=mean([Y(u,'semantic')-Y(u,'random') for u in r]);b=mean([Y(u,'full_semantic')-Y(u,'full_random') for u in r]);d[f'E_selected_F{f}_E{e}']=a;d[f'retention_F{f}_E{e}']=a-F(4,5)*b;d[f'E_full_F{f}_E{e}']=b
  d['Y_selected_sem_vs_native']=mean([Y(u,'semantic')-Y(u,'native') for u in rr]);d['G_selected_sem_vs_native']=mean([G(u,'semantic')-G(u,'native') for u in rr]);d['native_K2_accuracy']=F(sum(u['native']['scores']['2']['prediction']==u['correct_index'] for u in rr),32);d['native_K2_minus_K1_accuracy']=F(sum(int(u['native']['scores']['2']['prediction']==u['correct_index'])-int(u['native']['scores']['1']['prediction']==u['correct_index']) for u in rr),32)
  for lab in [0,1]:d[f'E_selected_label{lab}']=mean([Y(u,'semantic')-Y(u,'random') for u in rr if u['label_factor']==lab])
  group.append([float(d[n]) for n in names])
 a=np.array(group);assert a.shape==(64,18);rng=np.random.default_rng(2026111745);boots=np.empty((100000,18))
 for i in range(0,100000,2000):boots[i:i+2000]=a[rng.integers(0,64,(2000,64))].mean(axis=1)
 lo,hi=np.quantile(boots,[.05/36,1-.05/36],axis=0,method='linear');mu=a.mean(axis=0);stats={n:{'mean':float(mu[i]),'lower':float(lo[i]),'upper':float(hi[i])} for i,n in enumerate(names)};maxerr=max(abs(stats[n][k]-s['primary_18'][n][k]) for n in names for k in ['mean','lower','upper']);assert maxerr<2e-12;tests={n:stats[n]['lower']>.1 for n in names[:4]+names[8:14]+names[16:18]}
 for n in names[4:8]:tests[n]=stats[n]['lower']>=0
 tests['native_K2_accuracy_observed_atleast_0.70']=stats['native_K2_accuracy']['mean']>=.70;tests['native_K2_gain_lower_positive']=stats['native_K2_minus_K1_accuracy']['lower']>0
 points={f'F{f}_E{e}_Q{q}':float(mean([Y(u,'semantic')-Y(u,'random') for u in rows if (u['firsthop_factor'],u['endpoint_factor'],u['selector'])==(f,e,q)])) for f in [0,1] for e in [0,1] for q in [0,1]};assert points==s['eight_selector_points'];tests['all_eight_selector_positive']=all(v>0 for v in points.values())
 for name,sem,ran in [('selected_dose_QC','semantic','random'),('full_dose_QC','full_semantic','full_random')]:
  er,mass,m,n=dose(rows,sem,ran);tests[name]=er<=.01 and mass>=.99;p=s[name];assert p['pass']==tests[name] and abs(p['error_ratio']-er)<1e-12 and abs(p['matched_target_mass']-mass)<1e-12 and p['matched']==m and p['rows']==n
 assert tests==s['tests'];assert s['status']==('PASS' if all(tests.values()) else 'FAIL_FROZEN_CONFIRMATION_GATE');print('REGION_REPLAY_CONFIRM_PASS all18stats/CIs/gates; maxerror',maxerr);return {'status':'PASS','maximum_scalar_or_CI_error':maxerr,'primary_family':18,'bootstrap_replicates':100000,'independent_Fraction_group_means':True,'model_replication':False}
if __name__=='__main__':
 phase=sys.argv[1];out=dev() if phase=='development' else confirm();(R/('PUBLIC_REPLAY_'+phase.upper()+'.json')).write_text(json.dumps(out,indent=2))
