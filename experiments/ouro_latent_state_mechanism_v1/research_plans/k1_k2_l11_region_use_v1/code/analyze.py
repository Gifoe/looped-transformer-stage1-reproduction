import pathlib,json,numpy as np,sys,datetime
from guard import verify
R=pathlib.Path('/root/ouro-region-use-sol-20261117');D=R.parent/'ouro-binding-interface-sol-20261117';fs=verify()
def Y(u,k):
 x=u[k]['scores']['2']['candidate_logits'];return x[u['opposite_index']]-(x[u['decoy_indices'][0]]+x[u['decoy_indices'][1]])/2
def G(u,k):
 x=u[k]['scores']['2']['candidate_logits'];return x[u['opposite_index']]-x[u['correct_index']]
def qc(rows):
 t=[];a=[];matched=[]
 for u in rows:
  d=u['random']['audit']['events'][0]['dose'];assert d['target']==u['semantic']['audit']['events'][0]['actual_delta_norms'];t+=d['target'];a+=d['actual'];matched+=d['matched']
 t=np.array(t);a=np.array(a);err=abs(a-t);m=err<=np.maximum(1e-9,.01*t);assert m.tolist()==matched;total=t.sum();ratio=float(err.sum()/total) if total else 0.;mass=float(t[m].sum()/total) if total else 1.;return {'rows':len(t),'matched':int(m.sum()),'error_ratio':ratio,'matched_target_mass':mass,'pass':ratio<=.01 and mass>=.99}
def development():
 rows=[]
 for p in sorted((R/'results').glob('SR*_M*.json')):
  z=json.loads(p.read_text());assert z['freeze_sha256']==fs;rows+=z['rows']
 assert len(rows)==3072 and len({(u['id'],u['mask']) for u in rows})==3072
 full=[]
 for p in sorted((D/'results').glob('SR*_L11.json')):full+=json.loads(p.read_text())['rows']
 assert len(full)==512
 e7={f'F{f}_E{e}':float(np.mean([Y(u,'semantic')-Y(u,'random') for u in full if (u['firsthop_factor'],u['endpoint_factor'])==(f,e)])) for f in [0,1] for e in [0,1]};profiles={}
 for mask in range(1,8):
  rr=[u for u in rows if u['mask']==mask] if mask<7 else full;assert len(rr)==512;four={f'F{f}_E{e}':float(np.mean([Y(u,'semantic')-Y(u,'random') for u in rr if (u['firsthop_factor'],u['endpoint_factor'])==(f,e)])) for f in [0,1] for e in [0,1]};fine={f'F{f}_E{e}_Q{s}':float(np.mean([Y(u,'semantic')-Y(u,'random') for u in rr if (u['firsthop_factor'],u['endpoint_factor'],u['selector'])==(f,e,s)])) for f in [0,1] for e in [0,1] for s in [0,1]};dose=qc(rr);ratios={k:four[k]/e7[k] for k in four};eligible=all(v>.1 and v>=.8*e7[k] for k,v in four.items()) and all(v>0 for v in fine.values()) and dose['pass'];profiles[str(mask)]={'mask':mask,'open_regions':mask.bit_count(),'four_strata':four,'full_reference_four_strata':e7,'eight_selector_points':fine,'retention_ratios':ratios,'minimum_retention_ratio':min(ratios.values()),'dose_QC':dose,'eligible':eligible}
 candidates=[v for v in profiles.values() if v['eligible']];selected=min(candidates,key=lambda v:(v['open_regions'],-v['minimum_retention_ratio'],v['mask']))['mask'] if candidates else None
 out={'status':'SELECTED_FOR_FRESH_CONFIRMATION' if selected else 'STOP_NO_ELIGIBLE_MASK','selected_mask':selected,'profiles':profiles,'new_scientific_forwards':6144,'reused_full_mask_scientific_forwards':1024,'native_anchors_reused':512,'freeze_sha256':fs,'development_only':True,'goal_complete':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'REGION_SELECTION.json').write_text(json.dumps(out,indent=2));print('REGION_SELECTION',selected,{k:(p['minimum_retention_ratio'],p['eligible']) for k,p in profiles.items()},flush=True)
def confirmation():
 sel=json.loads((R/'REGION_SELECTION.json').read_text())['selected_mask'];assert sel is not None;rows=[]
 for p in sorted((R/'confirmation_results').glob('SR*.json')):
  z=json.loads(p.read_text());assert z['freeze_sha256']==fs and z['selected_mask']==sel;rows+=z['rows']
 assert len(rows)==2048 and len({u['id'] for u in rows})==2048
 names=[f'E_selected_F{f}_E{e}' for f in [0,1] for e in [0,1]]+[f'retention_F{f}_E{e}' for f in [0,1] for e in [0,1]]+[f'E_full_F{f}_E{e}' for f in [0,1] for e in [0,1]]+['Y_selected_sem_vs_native','G_selected_sem_vs_native','native_K2_accuracy','native_K2_minus_K1_accuracy','E_selected_label0','E_selected_label1'];assert len(names)==18;group=[]
 for pid in sorted({u['pair_id'] for u in rows}):
  rr=[u for u in rows if u['pair_id']==pid];assert len(rr)==32;d={}
  for f in [0,1]:
   for e in [0,1]:
    ss=[u for u in rr if (u['firsthop_factor'],u['endpoint_factor'])==(f,e)];a=float(np.mean([Y(u,'semantic')-Y(u,'random') for u in ss]));b=float(np.mean([Y(u,'full_semantic')-Y(u,'full_random') for u in ss]));d[f'E_selected_F{f}_E{e}']=a;d[f'retention_F{f}_E{e}']=a-.8*b;d[f'E_full_F{f}_E{e}']=b
  d['Y_selected_sem_vs_native']=float(np.mean([Y(u,'semantic')-Y(u,'native') for u in rr]));d['G_selected_sem_vs_native']=float(np.mean([G(u,'semantic')-G(u,'native') for u in rr]));d['native_K2_accuracy']=float(np.mean([u['native']['scores']['2']['prediction']==u['correct_index'] for u in rr]));d['native_K2_minus_K1_accuracy']=float(np.mean([int(u['native']['scores']['2']['prediction']==u['correct_index'])-int(u['native']['scores']['1']['prediction']==u['correct_index']) for u in rr]))
  for lab in [0,1]:d[f'E_selected_label{lab}']=float(np.mean([Y(u,'semantic')-Y(u,'random') for u in rr if u['label_factor']==lab]))
  group.append([d[n] for n in names])
 a=np.array(group);assert a.shape==(64,18);rng=np.random.default_rng(2026111745);boots=np.empty((100000,18))
 for start in range(0,100000,2000):boots[start:start+2000]=a[rng.integers(0,64,size=(2000,64))].mean(axis=1)
 p=.05/(2*18);lo,hi=np.quantile(boots,[p,1-p],axis=0,method='linear');means=a.mean(axis=0);stats={n:{'mean':float(means[j]),'lower':float(lo[j]),'upper':float(hi[j])} for j,n in enumerate(names)};pass_names=names[:4]+names[8:14]+names[16:18];tests={n:stats[n]['lower']>.1 for n in pass_names}
 for n in names[4:8]:tests[n]=stats[n]['lower']>=0
 tests['native_K2_accuracy_observed_atleast_0.70']=stats['native_K2_accuracy']['mean']>=.70;tests['native_K2_gain_lower_positive']=stats['native_K2_minus_K1_accuracy']['lower']>0
 points={f'F{f}_E{e}_Q{s}':float(np.mean([Y(u,'semantic')-Y(u,'random') for u in rows if (u['firsthop_factor'],u['endpoint_factor'],u['selector'])==(f,e,s)])) for f in [0,1] for e in [0,1] for s in [0,1]};tests['all_eight_selector_positive']=all(v>0 for v in points.values());dose=qc(rows);fullrows=[{**u,'semantic':u['full_semantic'],'random':u['full_random']} for u in rows];fullqc=qc(fullrows);tests['selected_dose_QC']=dose['pass'];tests['full_dose_QC']=fullqc['pass'];out={'status':'PASS' if all(tests.values()) else 'FAIL_FROZEN_CONFIRMATION_GATE','selected_mask':sel,'primary_18':stats,'tests':tests,'eight_selector_points':points,'selected_dose_QC':dose,'full_dose_QC':fullqc,'bootstrap':{'replicates':100000,'unit':'basegroup','groups':64,'seed':2026111745,'method':'linear','Bonferroni_family':18,'two_sided_simultaneous_confidence':.95},'selected7_retention_is_constructed_dependency':sel==7,'freeze_sha256':fs,'goal_complete':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'REGION_CONFIRMATION.json').write_text(json.dumps(out,indent=2));print('REGION_CONFIRMATION',out['status'],{k:v for k,v in tests.items() if not v},flush=True)
if __name__=='__main__':development() if sys.argv[1]=='development' else confirmation()
