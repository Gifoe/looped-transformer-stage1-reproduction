import pathlib,json,numpy as np,os,sys,datetime
R=pathlib.Path(os.environ.get('BRIDGE_ROOT','/root/ouro-temporal-bridge-sol-20261117'));phase=sys.argv[1];rows=[u for p in sorted((R/'B_results').glob('*.json')) for u in json.loads(p.read_text())['units'] if u['split']==phase] if not(R/'B_QUESTION_RESULTS.jsonl').exists() else [u for u in map(json.loads,(R/'B_QUESTION_RESULTS.jsonl').read_text().splitlines()) if u['split']==phase];assert len(rows)==(256 if phase=='development' else 512);groups=sorted({u['pair_id'] for u in rows});names=list(rows[0]['masks']);bg=rows[0]['background_g1']
def cluster(d):
 a=np.array([np.mean([v for u,v in zip(rows,d) if u['pair_id']==g]) for g in groups]);rng=np.random.default_rng(2026111711);boot=[]
 for _ in range(40):boot.extend(a[rng.integers(0,len(a),(250,len(a)))].mean(1).tolist())
 return {'mean':float(a.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'lower_bonferroni6':float(np.quantile(boot,.05/6)),'n':len(rows),'groups':len(groups)}
def margin(u,n):
 x=u['masks'][n]['scores']['2']['candidate_logits'];return x[u['answer1']]-x[u['answer0']]
values={n:np.array([margin(u,n) for u in rows]) for n in names};full=values['7']-values['0'];out={'phase':phase,'bg':bg,'full_vs_closed':cluster(full),'masks':{}}
for n,v in values.items():
 d=v-values['0'];strata={f'map{m}_dir{di}':float(np.mean([x for u,x in zip(rows,d) if u['mapping']==m and u['direction']==di])) for m in [0,1] for di in [0,1]};zs=[u['masks'][n]['scores']['2'] for u in rows];z={'correct':sum(a['prediction']==u['answer1'] for u,a in zip(rows,zs)),'W0_answers':sum(a['prediction']==u['answer0'] for u,a in zip(rows,zs)),'other_answers':sum(a['prediction'] not in [u['answer0'],u['answer1']] for u,a in zip(rows,zs)),'ties':sum(len(a['ties'])>1 for a in zs),'n':len(rows),'mean_m':float(v.mean()),'effect_vs_allclosed':cluster(d),'loss_vs_allopen':cluster(values['7']-v),'retention_80contrast':cluster(d-.8*full),'mapping_direction_means':strata,'positive_consistency':all(x>0 for x in strata.values()),'by_template':{str(t):float(np.mean([x for u,x in zip(rows,d) if u['template']==t])) for t in sorted({u['template'] for u in rows})}}
 z['retention_point']=float(d.mean()/full.mean()) if full.mean()!=0 else None;out['masks'][n]=z
(R/(phase+'_B_SUMMARY.json')).write_text(json.dumps(out,indent=2))
if phase=='development':
 eligible=[int(n) for n,z in out['masks'].items() if n not in ['0','7'] and z['effect_vs_allclosed']['lower_bonferroni6']>0 and z['retention_point']>=.8 and z['positive_consistency']];mask=min(eligible,key=lambda n:(n.bit_count(),-out['masks'][str(n)]['effect_vs_allclosed']['mean'],n)) if eligible else 7 if out['full_vs_closed']['ci95'][0]>0 else None
 (R/'B_SELECTION.json').write_text(json.dumps({'mask':mask,'regions':[x for bit,x in [(1,'G_map'),(2,'G_query'),(4,'G_read')] if mask is not None and mask&bit],'bg':bg,'eligible':eligible,'criterion':'Fewestregions retaining>=80%mean fullopen effect withBonferroni6positive andallmap/directionpositive; tieslargest effect thennumeric mask. Developmentonly.','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
else:
 selected=json.loads((R/'B_SELECTION.json').read_text())['mask'];z=out['masks'].get(str(selected));confirmed=z is not None and z['effect_vs_allclosed']['ci95'][0]>0 and z['positive_consistency'];(R/'B_CONFIRMATION_GATE.json').write_text(json.dumps({'pass':confirmed,'mask':selected,'bg':bg,'scope':'Onefrozenregionmask positivecluster95%effectvs0 withpositiveallmapping/directionmeans; noconfirmationreselection. Retentionandloss reported withoutthresholdchanges.','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
print('B_ANALYSIS',phase,{n:(z['effect_vs_allclosed']['mean'],z['retention_point']) for n,z in out['masks'].items()})
