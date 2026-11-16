import pathlib,json,numpy as np,os,sys,datetime
R=pathlib.Path(os.environ.get('BRIDGE_ROOT','/root/ouro-temporal-bridge-sol-20261117'));phase=sys.argv[1];rows=[u for p in sorted((R/'results').glob('TB*.json')) for u in json.loads(p.read_text())['units'] if u['split']==phase] if not (R/'ALL_QUESTION_RESULTS.jsonl').exists() else [u for u in map(json.loads,(R/'ALL_QUESTION_RESULTS.jsonl').read_text().splitlines()) if u['split']==phase];expected=256 if phase=='development' else 512;assert len(rows)==expected;groups=sorted({u['pair_id'] for u in rows})
def cluster(d):
 a=np.array([np.mean([v for u,v in zip(rows,d) if u['pair_id']==p]) for p in groups]);rng=np.random.default_rng(2026111711);v=[]
 for _ in range(40):v.extend(a[rng.integers(0,len(a),(250,len(a)))].mean(1).tolist())
 return {'mean':float(a.mean()),'ci95':np.quantile(v,[.025,.975]).tolist(),'bonferroni2_ci97_5':np.quantile(v,[.0125,.9875]).tolist(),'groups':len(groups),'n':len(rows)}
def margin(u,c,d='2'):
 z=u['conditions'][c]['scores'][d]['candidate_logits'];return z[u['answer1']]-z[u['answer0']]
values={c:np.array([margin(u,c) for u in rows]) for c in ['C00','C01','C10','C11']};contrasts={'capacity':values['C01']-values['C00'],'native':values['C11']-values['C10'],'K1_prior_contribution':values['C11']-values['C01'],'interaction':values['C11']-values['C10']-values['C01']+values['C00']};summary={'phase':phase,'n':len(rows),'groups':len(groups),'conditions':{},'contrasts':{},'native_text_baselines':{}}
for c in values:
 summary['conditions'][c]={}
 for dep in ['1','2']:
  zs=[u['conditions'][c]['scores'][dep] for u in rows];ms=np.array([margin(u,c,dep) for u in rows]);refm=np.array([u['native_W0']['scores'][dep]['candidate_logits'][u['answer1']]-u['native_W0']['scores'][dep]['candidate_logits'][u['answer0']] for u in rows]);newm=np.array([u['native_W1']['scores'][dep]['candidate_logits'][u['answer1']]-u['native_W1']['scores'][dep]['candidate_logits'][u['answer0']] for u in rows]);pred=np.array([z['prediction'] for z in zs]);a0=np.array([u['answer0'] for u in rows]);a1=np.array([u['answer1'] for u in rows]);unique=np.array([len(z['ties'])==1 for z in zs]);summary['conditions'][c][dep]={'W1_correct':int((pred==a1).sum()),'W0_answers':int((pred==a0).sum()),'other_answers':int(((pred!=a0)&(pred!=a1)).sum()),'n':len(rows),'ties':int((~unique).sum()),'strict_W1_unique_correct':int(((pred==a1)&unique).sum()),'mean_m':float(ms.mean()),'m_vs_native_W0':cluster(ms-refm),'m_vs_native_W1':cluster(ms-newm)}
for name,d in contrasts.items():
 strata={f'map{m}_dir{di}':float(np.mean([v for u,v in zip(rows,d) if u['mapping']==m and u['direction']==di])) for m in [0,1] for di in [0,1]};z=cluster(d);z['mapping_direction_means']=strata;z['positive_consistency']=all(v>0 for v in strata.values());z['pass']=(z['bonferroni2_ci97_5'][0]>0 and z['positive_consistency']) if name in ['capacity','native'] else None;z['by_template']={str(t):float(np.mean([v for u,v in zip(rows,d) if u['template']==t])) for t in sorted({u['template'] for u in rows})};z['by_desired_label']={str(l):float(np.mean([v for u,v in zip(rows,d) if u['answer1']==l])) for l in range(4)};summary['contrasts'][name]=z
for world in ['native_W0','native_W1']:
 summary['native_text_baselines'][world]={}
 for dep in ['1','2']:
  g='answer0' if world=='native_W0' else'answer1';summary['native_text_baselines'][world][dep]={'correct':sum(u[world]['scores'][dep]['prediction']==u[g] for u in rows),'n':len(rows),'ties':sum(len(u[world]['scores'][dep]['ties'])>1 for u in rows)}
 g='answer0' if world=='native_W0' else'answer1';summary['native_text_baselines'][world]['K1_wrong_K2_right']=sum(u[world]['scores']['1']['prediction']!=u[g] and u[world]['scores']['2']['prediction']==u[g] for u in rows);summary['native_text_baselines'][world]['K1_right_K2_wrong']=sum(u[world]['scores']['1']['prediction']==u[g] and u[world]['scores']['2']['prediction']!=u[g] for u in rows)
(R/(phase+'_A_SUMMARY.json')).write_text(json.dumps(summary,indent=2))
if phase=='development':
 background=1 if summary['contrasts']['native']['pass'] else 0 if summary['contrasts']['capacity']['pass'] else None;(R/'A_BACKGROUND_SELECTION.json').write_text(json.dumps({'g1':background,'basis':'native_reread_sensitivity' if background==1 else'delayed_capacity' if background==0 else'no_supported_A','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'chosen_on_development_only':True},indent=2))
print('TEMPORAL_A_SUMMARY',phase,{n:(z['mean'],z['bonferroni2_ci97_5'],z['pass']) for n,z in summary['contrasts'].items()})
