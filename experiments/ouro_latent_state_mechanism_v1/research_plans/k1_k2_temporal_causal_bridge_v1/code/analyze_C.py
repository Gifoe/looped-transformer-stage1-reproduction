import pathlib,json,numpy as np,os,sys,datetime
R=pathlib.Path(os.environ.get('BRIDGE_ROOT','/root/ouro-temporal-bridge-sol-20261117'));phase=sys.argv[1];rows=[u for p in sorted((R/'C_results').glob('*.json')) for u in json.loads(p.read_text())['units'] if u['split']==phase] if not(R/'C_QUESTION_RESULTS.jsonl').exists() else [u for u in map(json.loads,(R/'C_QUESTION_RESULTS.jsonl').read_text().splitlines()) if u['split']==phase];assert len(rows)==(256 if phase=='development' else 512);groups=sorted({u['pair_id'] for u in rows})
def cl(d):
 a=np.array([np.mean([v for u,v in zip(rows,d) if u['pair_id']==g]) for g in groups]);rng=np.random.default_rng(2026111711);v=[]
 for _ in range(40):v.extend(a[rng.integers(0,len(a),(250,len(a)))].mean(1).tolist())
 strata={f'map{m}_dir{di}':float(np.mean([v for u,v in zip(rows,d) if u['mapping']==m and u['direction']==di])) for m in [0,1] for di in [0,1]};ci=np.quantile(v,[.025/7,1-.025/7]).tolist();return {'mean':float(a.mean()),'ci95':np.quantile(v,[.025,.975]).tolist(),'bonferroni7_ci':ci,'n':len(rows),'groups':len(groups),'mapping_direction_means':strata,'positive_consistency':all(v>0 for v in strata.values()),'pass':ci[0]>0 and all(v>0 for v in strata.values())}
def margin(u,c):
 x=u['conditions'][c]['scores']['2']['candidate_logits'];return x[u['answer1']]-x[u['answer0']]
v={c:np.array([margin(u,c) for u in rows]) for c in rows[0]['conditions']};out={'phase':phase,'n':len(rows),'groups':len(groups),'conditions':{},'primary_contrasts':{},'consistency_identities':{}}
for c,m in v.items():
 z=[u['conditions'][c]['scores']['2'] for u in rows];out['conditions'][c]={'W1_correct':sum(a['prediction']==u['answer1'] for u,a in zip(rows,z)),'W0_answers':sum(a['prediction']==u['answer0'] for u,a in zip(rows,z)),'other_answers':sum(a['prediction'] not in [u['answer0'],u['answer1']] for u,a in zip(rows,z)),'ties':sum(len(a['ties'])>1 for a in z),'strict_W1_unique_correct':sum(a['prediction']==u['answer1'] and len(a['ties'])==1 for u,a in zip(rows,z)),'mean_m':float(m.mean()),'vs_Z_descriptive':cl(m-v['Z'])}
for name in ['early','mid','late']:
 out['primary_contrasts']['restore_'+name+'_vs_block']=cl(v['restore_'+name]-v['block']);out['primary_contrasts']['T_vs_reverse_'+name]=cl(v['T']-v['reverse_'+name])
out['primary_contrasts']['T_vs_matched_random']=cl(v['T']-v['random']);out['consistency_identities']['block_equals_Z_full49152']=all(u['conditions']['block']['scores'][d]['full49152_sha256']==u['conditions']['Z']['scores'][d]['full49152_sha256'] for u in rows for d in ['1','2']);assert out['consistency_identities']['block_equals_Z_full49152'];out['dose']={'sites':0,'matched':0,'unmatched_retained':[],'max_relative_error':0.}
for u in rows:
 for e in u['conditions']['random']['cut']['path_edits']:
  dose=e['dose']
  for i,(t,a,er,ok) in enumerate(zip(dose['target'],dose['actual'],dose['relative_error'],dose['matched'])):
   out['dose']['sites']+=1;out['dose']['matched']+=ok;out['dose']['max_relative_error']=max(er,out['dose']['max_relative_error'])
   if not ok:out['dose']['unmatched_retained'].append({'unit':u['id'],'layer':e['layer'],'receiver_position':e['receiver_span'][0]+i,'target':t,'actual':a,'relative_error':er})
(R/(phase+'_C_SUMMARY.json')).write_text(json.dumps(out,indent=2));print('C_ANALYSIS',phase,{n:(z['mean'],z['bonferroni7_ci'],z['pass']) for n,z in out['primary_contrasts'].items()})
