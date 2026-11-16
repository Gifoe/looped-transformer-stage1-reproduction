from freeze_guard import verify,exact_ids
FREEZE_SHA=verify()
"""All-world base-group inference, no native-correct filtering."""
import pathlib,json,numpy as np,sys,os
def load_rows(folder,phase,qs):
 result=[]
 for path in sorted((R/folder).glob('SR*.json')):
  z=json.loads(path.read_text())
  if z['split']!=phase:continue
  assert z['freeze_sha256']==FREEZE_SHA
  exact_ids(z['rows'],[q['id'] for q in qs.values() if q['pair_id']==z['pair_id']])
  result.extend(z['rows'])
 return result
R=pathlib.Path(os.environ.get('SEMANTIC_ROOT','/root/ouro-semantic-read-sol-20261117'));phase=sys.argv[1];stage=sys.argv[2];qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};nr=load_rows('native_results',phase,qs);N=512 if phase=='development' else 2048;exact_ids(nr,[q['id'] for q in qs.values() if q['split']==phase]);assert len(nr)==N;groups=sorted({qs[u['id']]['pair_id'] for u in nr});rng=np.random.default_rng(2026111733);indices=rng.integers(0,len(groups),(10000,len(groups)))
def ci_by_group(data,alpha=.025):
 a=np.array(data,dtype=float);bs=a[indices].mean(1);return {'mean':float(a.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist(),'adjusted_bonferroni7':np.quantile(bs,[alpha/7,1-alpha/7]).tolist(),'groups':len(groups),'bootstrap':10000}
acc={str(loop):sum(u['native']['scores'][str(loop)]['prediction']==qs[u['id']]['correct_index'] for u in nr) for loop in [1,2]};delta=[np.mean([int(u['native']['scores']['2']['prediction']==qs[u['id']]['correct_index'])-int(u['native']['scores']['1']['prediction']==qs[u['id']]['correct_index']) for u in nr if qs[u['id']]['pair_id']==g]) for g in groups];diff=ci_by_group(delta);cap={'freeze_sha256':FREEZE_SHA,'phase':phase,'worlds':N,'groups':len(groups),'native_correct':acc,'native_K2_accuracy':acc['2']/N,'K2_minus_K1_accuracy':diff,'pass':acc['2']/N>=.70 and diff['ci95'][0]>0};(R/(phase+'_NATIVE_CAPABILITY.json')).write_text(json.dumps(cap,indent=2));print('NATIVE_CAPABILITY',phase,cap['pass'],acc,diff['ci95'])
if stage=='native':sys.exit(0)
rows=load_rows('results',phase,qs);exact_ids(rows,[q['id'] for q in qs.values() if q['split']==phase]);assert len(rows)==N

def score(u,condition,loop,metric):
 x=u['conditions'][condition]['scores'][str(loop)]['candidate_logits'];return x[u['opposite_index']]-(np.mean([x[k] for k in u['decoy_indices']]) if metric=='Y' else x[u['correct_index']])
def values(u):
 y2s=score(u,'semanticK2',2,'Y');y2r=score(u,'randomK2',2,'Y');y1s=score(u,'semanticK1',1,'Y');y1r=score(u,'randomK1',1,'Y');return [y2s-y2r,y2s-score(u,'native',2,'Y'),score(u,'semanticK2',2,'G')-score(u,'native',2,'G'),y2s-y2r-y1s+y1r]
v={u['id']:values(u) for u in rows};primary={}
for f in [0,1]:
 for e in [0,1]:
  z=ci_by_group([np.mean([v[u['id']][0] for u in rows if u['pair_id']==g and u['firsthop_factor']==f and u['endpoint_factor']==e]) for g in groups]);z['pass']=z['adjusted_bonferroni7'][0]>.1;primary[f'specific_F{f}_E{e}']=z
for name,col,threshold in [('Y_semantic_vs_native',1,.1),('G_semantic_vs_native',2,.1),('round_specificity_difference_K2_minus_K1',3,0.)]:
 z=ci_by_group([np.mean([v[u['id']][col] for u in rows if u['pair_id']==g]) for g in groups]);z['pass']=z['adjusted_bonferroni7'][0]>threshold;primary[name]=z
strata={f'F{f}_E{e}_Q{s}':float(np.mean([v[u['id']][0] for u in rows if u['firsthop_factor']==f and u['endpoint_factor']==e and u['selector']==s])) for f in [0,1] for e in [0,1] for s in [0,1]};dose={'all':[],'K1':[],'K2':[]};bylayer={}
for u in rows:
 for arm in [1,2]:
  sem={e['layer']:e for e in u['conditions']['semanticK'+str(arm)]['audit']['events']}
  for e in u['conditions']['randomK'+str(arm)]['audit']['events']:
   d=e['dose'];assert d['target']==sem[e['layer']]['actual_delta_norms'];z=list(zip(d['target'],d['actual'],d['matched']));dose['all'].extend(z);dose['K'+str(arm)].extend(z);bylayer.setdefault(f'K{arm}_L{e["layer"]}',[]).extend(z)
def dose_stats(rows):
 a=np.array([[t,r] for t,r,ok in rows]);t=a[:,0];actual=a[:,1];err=np.abs(actual-t);matched=err<=np.maximum(.01*t,1e-9);assert matched.tolist()==[ok for t,r,ok in rows];mass=float(t.sum());errratio=float(err.sum()/mass) if mass else 0.;fraction=float(t[matched].sum()/mass) if mass else None;return {'rows':len(rows),'matched_rows':int(matched.sum()),'unmatched_rows_retained':int((~matched).sum()),'target_norm_sum':mass,'absolute_error_div_target_sum':errratio,'matched_target_mass_fraction':fraction,'degenerate_all_zero':mass==0,'actual_target_norm_sum_ratio':float(actual.sum()/mass) if mass else None,'max_relative_error':float(np.max(err/np.maximum(t,1e-30))),'pass':bool((errratio<=.01 and fraction>=.99) if mass else np.all(actual==0))}
qc={k:dose_stats(z) for k,z in dose.items()};layers={k:dose_stats(z) for k,z in bylayer.items()};conditions={}
for name in rows[0]['conditions']:
 conditions[name]={}
 for loop in [1,2]:
  zz=[u['conditions'][name]['scores'][str(loop)] for u in rows];counts={'selected':0,'opposite':0,'decoy':0};sums={k:[] for k in ['selected','opposite','decoy1','decoy2']}
  for u,z in zip(rows,zz):
   p=z['prediction'];counts['selected' if p==u['correct_index'] else'opposite' if p==u['opposite_index'] else'decoy']+=1;x=z['candidate_logits']
   for role,index in [('selected',u['correct_index']),('opposite',u['opposite_index']),('decoy1',u['decoy_indices'][0]),('decoy2',u['decoy_indices'][1])]:sums[role].append(x[index])
  conditions[name][str(loop)]={'counts':counts,'ties':sum(len(z['ties'])>1 for z in zz),'mean_role_logits':{k:float(np.mean(z)) for k,z in sums.items()},'mean_Y':float(np.mean([score(u,name,loop,'Y') for u in rows])),'mean_G':float(np.mean([score(u,name,loop,'G') for u in rows]))}
positive=all(x>0 for x in strata.values());gate=cap['pass'] and all(z['pass'] for z in qc.values()) and all(z['pass'] for k,z in primary.items() if not k.startswith('round_')) and positive;out={'freeze_sha256':FREEZE_SHA,'phase':phase,'worlds':N,'base_groups':len(groups),'primary_contrasts':primary,'eight_semantic_strata':strata,'all_eight_point_directions_positive':positive,'conditions':conditions,'dose_QC':qc,'dose_by_layer':layers,'graph_directed_pairmessage_gate':gate,'stronger_K2_participation':gate and primary['round_specificity_difference_K2_minus_K1']['pass'],'native_capability':cap,'scope':'idealpairedmessageintervention notnativeKVrecompute; notG-localcomposition/uniquealgorithm; allworldsandt iesretained'};(R/(phase+'_SEMANTIC_SUMMARY.json')).write_text(json.dumps(out,indent=2));print('SEMANTIC_GATE',phase,gate,'strongerK2',out['stronger_K2_participation']);print({k:(z['mean'],z['adjusted_bonferroni7'],z['pass']) for k,z in primary.items()});print('DOSE_QC',qc)
