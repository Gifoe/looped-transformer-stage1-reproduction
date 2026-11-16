"""Frozen directional diagnosis; gold is used offline, never by forward control."""
import pathlib,json,numpy as np,datetime,hashlib,time
R=pathlib.Path('/root/ouro-catc-critical-sol-20261116');O=pathlib.Path('/root/ouro-catc-sol-20261116');spec=json.loads((R/'FREEZE.json').read_text());assert json.loads((R/'TECHNICAL_GATE.json').read_text())['status']=='PASS';start=time.monotonic();names={'Native':'T0','V+':'T1_structured','V-':'T3_structured','P+':'T2_structured','P-':'T4_structured'};cohort=spec['cohorts']['development'];allrows={q['id']:q for q in map(json.loads,(O/'TRAINING_FREE_ALL_ROWS.jsonl').read_text().splitlines())}
def ce(v,g):
 x=np.array(v,dtype=np.float64);a=x.max();return float(a+np.log(np.exp(x-a).sum())-x[g])
def margin(v,g):return float(v[g]-max(x for i,x in enumerate(v) if i!=g))
def preference(loss,base):
 a,b=loss
 if min(a,b)>=base-1e-8 or abs(a-b)<=1e-8:return 'neutral_or_tie'
 return 'positive' if a<b else 'negative'
rows=[]
for id in cohort:
 q=allrows[id];g=q['gold'];records={v['name']:v for v in q['records']};methods={n:records[old] for n,old in names.items()};row={'id':id,'gold':g,'methods':methods,'offline_diagnostics':{}}
 for d in ['2','4']:
  losses={n:ce(v['scores'][d]['candidate_logits'],g) for n,v in methods.items()};pref={c:preference([losses[c+'+'],losses[c+'-']],losses['Native']) for c in ['V','P']};row['offline_diagnostics'][d]={'CE':losses,'CE_change_vs_native':{n:v-losses['Native'] for n,v in losses.items()},'preference':pref,'oracle_min_CE':min(losses.values()),'oracle_any_correct':any(v['scores'][d]['prediction']==g for v in methods.values()),'native_correct':methods['Native']['scores'][d]['prediction']==g}
 rows.append(row)
def summarize(rs):
 out={}
 for d in ['2','4']:
  gold=np.array([q['gold'] for q in rs]);native=np.array([q['methods']['Native']['scores'][d]['prediction']==q['gold'] for q in rs]);methods={}
  for n in names:
   correct=np.array([q['methods'][n]['scores'][d]['prediction']==q['gold'] for q in rs]);changes=np.array([q['offline_diagnostics'][d]['CE_change_vs_native'][n] for q in rs]);methods[n]={'correct':int(correct.sum()),'n':len(rs),'mean_candidate_CE':float(np.mean([q['offline_diagnostics'][d]['CE'][n] for q in rs])),'mean_CE_change_vs_same_depth_native':float(changes.mean()),'CE_benefit_positive_negative_neutral':[int((changes < -1e-8).sum()),int((changes>1e-8).sum()),int((abs(changes)<=1e-8).sum())],'same_depth_rescue':int((correct&~native).sum()),'same_depth_damage':int((~correct&native).sum())}
  out[d]={'methods':methods,'preference':{c:{v:sum(q['offline_diagnostics'][d]['preference'][c]==v for q in rs) for v in ['positive','negative','neutral_or_tie']} for c in ['V','P']},'gold_oracle_any_correct':sum(q['offline_diagnostics'][d]['oracle_any_correct'] for q in rs),'gold_oracle_mean_min_CE':float(np.mean([q['offline_diagnostics'][d]['oracle_min_CE'] for q in rs]))}
 return out
summary={'full_development':summarize(rows),'halves':{str(i):summarize(rows[i*128:(i+1)*128]) for i in range(2)},'quarters':{str(i):summarize(rows[i*64:(i+1)*64]) for i in range(4)}};loss=summary['full_development']['4']['methods'];choice=min(['V+','V-','P+','P-'],key=lambda n:(loss[n]['mean_candidate_CE'],['V+','V-','P+','P-'].index(n)));selected={'name':choice,'original_record':names[choice],'coefficients':spec['settings'][choice],'physical_layers':[10,11,12,13,14,15,16],'recurrent_pass':2,'readout':'R','criterion':spec['selection'],'development_n':len(rows),'development_mean_K4_CE':loss[choice]['mean_candidate_CE'],'native_development_mean_K4_CE':loss['Native']['mean_candidate_CE'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()};(R/'SELECTED_METHOD.json').write_text(json.dumps(selected,indent=2));(R/'DIRECTIONAL_SUMMARY.json').write_text(json.dumps(summary,indent=2));(R/'DEVELOPMENT_ROWS.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows));(R/'EXPERIMENT1_COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','n':len(rows),'new_GPU_forwards':0,'wall_seconds':time.monotonic()-start,'selected':choice,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));print('DIRECTIONAL_COMPLETE',choice,json.dumps(summary['full_development']))
