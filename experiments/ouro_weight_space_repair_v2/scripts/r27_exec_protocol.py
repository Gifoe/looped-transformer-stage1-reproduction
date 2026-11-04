"""Fixed synthetic task gates; no ARC readers or estimator selection."""
PATCHES=[('V_identity','V','B',False),('V_clean','V','A',False),('V_shuffle','V','A',True),('K_clean','K','A',False),('K_shuffle','K','A',True),('K_identity','K','B',False)]
def ordered_instances(manifest):
 assert manifest['status']=='passed_CPU_structural_admission_only' and len(manifest['tables'])==2
 out=[]
 for table in manifest['tables']:
  assert len(table['instances'])==2
  for q in table['instances']:
   for v in q['variants']:out.append(dict(table=table['table'],query_row=q['query_row'],**v))
 assert len(out)==8 and [v['version'] for v in out]==['A','B']*4
 return out

def label_metrics(z,label_ids,expected):
 import numpy as np
 z=np.asarray(z,dtype=np.float64).reshape(-1);assert z.size==49152 and np.isfinite(z).all();assert len(set(label_ids.values()))==4
 maximum=z.max();log_partition=maximum+np.log(np.exp(z-maximum).sum());lp={k:float(z[v]-log_partition) for k,v in label_ids.items()}
 strict=all(z[label_ids[expected]]>z[label_ids[k]] for k in label_ids if k!=expected)
 return dict(label_logits={k:float(z[v]) for k,v in label_ids.items()},label_log_probabilities=lp,label_probability_mass=float(sum(np.exp(v) for v in lp.values())),expected_strictly_highest=bool(strict))
def functional_summary(baselines,patches):
 # Called only after complete, clean all-numeric checks and fixed Phase A/B schema.
 phaseA=all(b['metrics']['expected_strictly_highest'] for b in baselines)
 if not phaseA:
  assert not patches;return dict(phaseA_pass=False,phaseB_performed=False,scope='Four-label subspace screen failed; not a general ICL impossibility claim')
 assert len(patches)==24
 outcomes={}
 for kind in ['V','K']:
  rows=[]
  for i in range(4):
   a,b=baselines[2*i:2*i+2];ya=a['expected_label'];yb=b['expected_label'];assert ya!=yb
   def margin(r):p=r['metrics']['label_logits'];return p[ya]-p[yb]
   group=patches[i*6:(i+1)*6];clean=next(r for r in group if r['patch']==kind+'_clean');shuffle=next(r for r in group if r['patch']==kind+'_shuffle')
   mb=margin(b);mc=margin(clean);ms=margin(shuffle);rows.append(dict(instance=i,baseline_B_margin=mb,clean_margin=mc,shuffle_margin=ms,transport_delta=mc-mb,specificity=mc-ms,crossed_zero=mc>0))
  passed=all(r['transport_delta']>0 and r['specificity']>0 for r in rows) and sum(r['crossed_zero'] for r in rows)>=3
  outcomes[kind]=dict(rows=rows,fixed_screen_pass=passed,role='primary functional diagnostic' if kind=='V' else 'independent descriptive path; cannot substitute for V')
 return dict(phaseA_pass=True,phaseB_performed=True,paths=outcomes,scope='Finite four-label subspace diagnostic only, no ARC benefit or deployment authorization')

def execution_schema(raw,rows,phaseB,projection_width):
 assert len(rows)==8 and len(raw['baselines'])==8 and raw['no_op_exact'] and not raw['invalid_reasons']
 def same_input(record,row):
  for field in ['table','query_row','version','input_ids_sha256','input_shape','source_positions','expected_label']:assert record[field]==row[field]
  assert record['label_ids']=={x['label']:x['token_id'] for x in row['sources']}
  assert len(record['query_states'])==4 and all(r['shape']==[1,1,2048] and r['dtype']=='torch.bfloat16' for r in record['query_states']) and record['fullV']['shape']==[1,1,49152] and record['fullV']['dtype']=='torch.float32'
 def clock(c,row,mode):
  assert c['mode']==mode and c['calls_per_module']==[4]*48 and c['source_positions']==row['source_positions']
  expected=[[loop,layer,kind] for loop in range(4) for layer in range(24) for kind in ['K','V']];assert c['events']==expected
 for r,row in zip(raw['baselines'],rows):same_input(r,row);clock(r['clock'],row,'capture')
 same_input(raw['no_hook'],rows[0]);assert raw['no_hook']['query_states']==raw['baselines'][0]['query_states'] and raw['no_hook']['fullV']==raw['baselines'][0]['fullV']
 pairs=raw['source_pairs'];expected={(t,v,k,l) for t in [1,2] for v in ['A','B'] for k in ['K','V'] for l in range(24)};keys=[(p['table'],p['version'],p['kind'],p['layer']) for p in pairs];assert len(keys)==len(set(keys))==192 and set(keys)==expected
 for p in pairs:
  a,b=[r for r in raw['baselines'] if r['table']==p['table'] and r['version']==p['version']];assert p['A']==a['donors'][p['kind']][p['layer']] and p['B']==b['donors'][p['kind']][p['layer']] and p['A']==p['B'] and p['exact']
 for r in raw['baselines']:
  assert set(r['donors'])=={'K','V'} and all(len(r['donors'][k])==24 for k in ['K','V'])
  assert all(v['dtype']=='torch.bfloat16' and v['shape']==[1,4,projection_width] for vv in r['donors'].values() for v in vv)
  if r['version']=='B':assert len(r['first3_full_states'])==3 and all(v['shape']==[1,r['input_shape'][1],2048] and v['dtype']=='torch.bfloat16' for v in r['first3_full_states'])
 expected_count=33 if phaseB else 9;assert len(raw['forward_records'])==expected_count and [r['number'] for r in raw['forward_records']]==list(range(1,expected_count+1))
 assert raw['forward_records'][0]['phase']=='no_hook'
 for call,base in zip(raw['forward_records'][1:9],raw['baselines']):assert call['phase']=='A' and all(call[k]==base[k] for k in ['table','query_row','version','clock'])
 assert len(raw['patches'])==(24 if phaseB else 0)
 if phaseB:
  for i,row in enumerate([r for r in rows if r['version']=='B']):
   base=raw['baselines'][2*i+1];firstquery=next(r['query_row'] for r in rows if r['table']==row['table'])
   for j,(name,kind,version,shuffle) in enumerate(PATCHES):
    r=raw['patches'][i*6+j];same_input(r,row);clock(r['clock'],row,'patch');assert r['patch']==name and r['clock']['kind']==kind and r['clock']['other_rows_exact_verified']
    assert r['clock']['permutation']==([1,2,3,0] if shuffle else [0,1,2,3])
    assert (r['donor_table'],r['donor_query_row'],r['donor_version'],r['donor_kind'])==(row['table'],firstquery,version,kind)
    donor=next(b for b in raw['baselines'] if b['table']==row['table'] and b['query_row']==firstquery and b['version']==version);assert r['donor_vectors']==donor['donors'][kind]
    assert r['first3_full_exact'] and r['first3_full_states']==r['baseline_first3_full_states']==base['first3_full_states']
    if name.endswith('identity'):assert r['identity_exact'] and r['query_states']==base['query_states'] and r['fullV']==base['fullV']
    call=raw['forward_records'][9+i*6+j];assert call['phase']=='B' and all(call[k]==r[k] for k in ['table','query_row','patch','clock'])
 assert raw['cleanup']['handles_removed'] and raw['cleanup']['original_hooks_restored'] and raw['cleanup']['all_model_hooks_empty'] and not raw['cleanup']['errors']
 return expected_count

def check_resources(sup,book,inherited_values):
 assert sup['status']=='complete' and sup['exit_code']==0 and not sup.get('timed_out',False) and not sup['overrun'] and 0<sup['gpu_worker_seconds']<=180
 assert len(inherited_values)==15 and sum(inherited_values)==18874.135196831077
 inherited=sum(inherited_values);assert book['used_seconds']==sup['gpu_worker_seconds'] and book['inherited_total_seconds']==inherited and book['total_used_seconds']==inherited+book['used_seconds']<=21600
 return book['used_seconds']
