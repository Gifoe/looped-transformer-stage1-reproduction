"""Audit and map exposed ARC test historical endpoints; never infer K2/K3."""
import csv,collections,json,hashlib,math,tarfile,io,sys,itertools
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
P=Path(__file__).resolve();R=P.parents[2];ROOT=R.parent
sys.path.insert(0,str(R/'experiments/ouro_weight_space_repair_v2/scripts'))
from r10_core import joint_prefix,joint_requests
from r7_core import digest
source=R/'experiments/ouro_weight_space_repair_v2/recurrent_decision_audit_v1/loopwise_predictions.csv';data=ROOT/'data/test.parquet';rows=pq.read_table(data).to_pylist();docs={r['id']:r for r in rows}
if len(rows)!=1172 or len(docs)!=1172:raise RuntimeError('bad official ID coverage')
table={}
with source.open(encoding='utf8',newline='') as f:
 for row in csv.DictReader(f):
  key=(row['id'],row['family'],int(row['K']))
  if key in table:raise RuntimeError('duplicate record '+repr(key))
  if not row['id'].startswith('arc_challenge/test/'):raise RuntimeError('bad ID namespace')
  rawid=row['id'].split('/')[-1];doc=docs.get(rawid)
  if doc is None:raise RuntimeError('unknown question ID')
  labels=doc['choices']['label'];gold=int(row['gold']);scores=json.loads(row['scores']);pred=int(row['prediction']);correct=row['correct']=='True'
  if gold!=labels.index(doc['answerKey']) or len(scores)!=len(labels) or not all(math.isfinite(x) for x in scores):raise RuntimeError('gold, option count or score validity mismatch')
  rank=sorted(range(len(scores)),key=lambda j:-scores[j]);wrong=max((j for j in range(len(scores)) if j!=gold),key=lambda j:scores[j]);margin=scores[gold]-scores[wrong]
  if pred!=rank[0] or correct!=(pred==gold) or not math.isclose(float(row['gold_margin']),margin,rel_tol=1e-6,abs_tol=1e-6):raise RuntimeError('prediction/correctness/margin mismatch')
  table[key]={'scores':scores,'prediction':pred,'gold':gold,'correct':correct,'rank':rank,'gold_rank':rank.index(gold)+1,'strongest_wrong':wrong,'gold_margin':margin,'top_tie':sum(x==max(scores) for x in scores)>1,'any_tie':len(set(scores))<len(scores)}
ids=sorted('arc_challenge/test/'+r['id'] for r in rows);expected={(i,f,k) for i in ids for f in ('joint','cyclic_mean') for k in (1,4)}
if len(table)!=len(expected) or set(table)!=expected:raise RuntimeError('incomplete or unexpected protocol-depth coverage')
archive_root=R/'experiments/ouro_weight_space_repair_v2';archive_index=json.loads((archive_root/'artifacts/r11_RECORDS_INDEX.json').read_text());parts=[]
for item in archive_index['parts']:
 payload=(archive_root/item['path']).read_bytes()
 if len(payload)!=item['bytes'] or hashlib.sha256(payload).hexdigest()!=item['sha256']:raise RuntimeError('archive part integrity failure')
 parts.append(payload)
bundle=b''.join(parts)
if hashlib.sha256(bundle).hexdigest()!=archive_index['archive_sha256']:raise RuntimeError('archive integrity failure')
with tarfile.open(fileobj=io.BytesIO(bundle),mode='r:xz') as archive:
 source_raw=archive.extractfile('results/r11_test_selected.jsonl').read()
if hashlib.sha256(source_raw).hexdigest()!='93845afb6a449590e465dac8ed8430e22ee618e64559f19a8fec1ff805eff9da':raise RuntimeError('historical raw scores integrity failure')
raw_records=[json.loads(line) for line in source_raw.splitlines()];raw_by_id={r['id']:r for r in raw_records}
if len(raw_records)!=1172 or len(raw_by_id)!=1172 or set(raw_by_id)!=set(ids):raise RuntimeError('raw historical ID coverage failure')
demo_path=R/'experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json'
if hashlib.sha256(demo_path.read_bytes()).hexdigest()!='8cf4b70e7c07c67a589560d5fcbc1fa8feb85a0e7af3b348fa21f147271419da':raise RuntimeError('demonstration source hash mismatch')
prefix=joint_prefix(json.loads(demo_path.read_text(encoding='utf8')));leave_out={}
for ident in ids:
 raw=raw_by_id[ident];doc=docs[ident.split('/')[-1]];choices=doc['choices']['text'];n=len(choices);rot=raw['cyclic_rotations']
 if len(rot)!=n or raw['actual_auxiliary_depth']!=4 or len(raw['target_token_counts'])!=n*n or any(x!=1 for x in raw['target_token_counts']):raise RuntimeError('rotation count/depth/target mismatch')
 for shift,record in enumerate(rot):
  order=[(j+shift)%n for j in range(n)]
  if record['order']!=order:raise RuntimeError('rotation order mismatch')
  expected_requests=joint_requests({'question':doc['question'],'choices':[choices[j] for j in order]},prefix)
  if [digest(q) for q in expected_requests]!=record['request_sha256']:raise RuntimeError('rotation request provenance mismatch')
 for k in (1,4):
  matrix=np.asarray([record['sum_scores'][f'native_k{k}'] for record in rot],dtype=np.float64);original=np.asarray(raw['joint_baseline_scores'][f'joint{k}_only']);pooled=np.asarray(raw['joint_baseline_scores'][f'cyclic_joint{k}_only'])
  if matrix.shape!=(n,n) or not np.isfinite(matrix).all() or not np.array_equal(matrix[0],original) or not np.array_equal(matrix[0],np.asarray(raw['auxiliary_sum_scores'][f'native_k{k}'])) or not np.array_equal(matrix.mean(axis=0),pooled):raise RuntimeError('cyclic pooling provenance mismatch')
  without=matrix[1:].mean(axis=0);reconstructed=(n*pooled-original)/(n-1)
  if not np.array_equal(without,reconstructed):raise RuntimeError('leave-original-out reconstruction mismatch')
  leave_out[(ident,k)]={'scores':without.tolist(),'prediction':int(np.argmax(without)),'correct':int(np.argmax(without))==raw['gold']}
 for family,keyprefix in (('joint','joint'),('cyclic_mean','cyclic_joint')):
  for k in (1,4):
   if raw['joint_baseline_scores'][f'{keyprefix}{k}_only']!=table[ident,family,k]['scores'] or raw['gold']!=table[ident,family,k]['gold']:raise RuntimeError('derived score does not match historical raw input')
counts={str(k):collections.Counter() for k in (1,4)};matrix=collections.Counter();families={f:collections.Counter() for f in ('joint','cyclic_mean')};agreement=collections.Counter();switches=collections.Counter();strata={f:collections.defaultdict(collections.Counter) for f in families};agreement_outcomes=collections.Counter();rank_strata={f:collections.defaultdict(collections.Counter) for f in families};case_ids=collections.defaultdict(list);leave_counts={str(k):collections.Counter() for k in (1,4)};items=[]
for ident in ids:
 endpoints={str(k):{f:table[ident,f,k] for f in families} for k in (1,4)};raw=ident.split('/')[-1];option_count=len(docs[raw]['choices']['label'])
 for k in (1,4):
  a,b=(table[ident,f,k] for f in ('joint','cyclic_mean'))
  if a['gold']!=b['gold']:raise RuntimeError('gold differs across views')
  counts[str(k)][('same' if a['prediction']==b['prediction'] else 'different')+'|joint_'+str(a['correct']).lower()+'|cyclic_'+str(b['correct']).lower()]+=1
 same1=table[ident,'joint',1]['prediction']==table[ident,'cyclic_mean',1]['prediction'];same4=table[ident,'joint',4]['prediction']==table[ident,'cyclic_mean',4]['prediction'];agreement[f'K1_{"same" if same1 else "different"}|K4_{"same" if same4 else "different"}']+=1
 a1=table[ident,'joint',1];b1=table[ident,'cyclic_mean',1];a4=table[ident,'joint',4];b4=table[ident,'cyclic_mean',4]
 if not same1 and same4:agreement_outcomes['different_to_same|start_joint_'+str(a1['correct']).lower()+'|start_cyclic_'+str(b1['correct']).lower()+'|end_correct_'+str(a4['correct']).lower()]+=1
 if same1 and not same4:agreement_outcomes['same_to_different|start_correct_'+str(a1['correct']).lower()+'|end_joint_'+str(a4['correct']).lower()+'|end_cyclic_'+str(b4['correct']).lower()]+=1;case_ids['same_to_different'].append(ident)
 for k in (1,4):
  z=leave_out[(ident,k)];j=table[ident,'joint',k];c=table[ident,'cyclic_mean',k]
  leave_counts[str(k)]['leave_correct']+=z['correct'];leave_counts[str(k)]['leave_same_joint']+=z['prediction']==j['prediction'];leave_counts[str(k)]['leave_same_pooled']+=z['prediction']==c['prediction'];leave_counts[str(k)]['leave_differs_pooled']+=z['prediction']!=c['prediction']
 kinds={}
 for f in families:
  a,b=table[ident,f,1],table[ident,f,4];kinds[f]=('correct' if a['correct'] else 'wrong')+'->'+('correct' if b['correct'] else 'wrong');families[f][kinds[f]]+=1
  families[f]['same_top']+=a['prediction']==b['prediction'];families[f]['full_rank_changed']+=a['rank']!=b['rank'];families[f]['same_top_rank_changed']+=a['prediction']==b['prediction'] and a['rank']!=b['rank'];families[f]['gold_rank_improved']+=b['gold_rank']<a['gold_rank'];families[f]['gold_rank_same']+=b['gold_rank']==a['gold_rank'];families[f]['gold_rank_worsened']+=b['gold_rank']>a['gold_rank'];families[f]['strongest_wrong_changed']+=a['strongest_wrong']!=b['strongest_wrong'];families[f]['K1_top_ties']+=a['top_tie'];families[f]['K4_top_ties']+=b['top_tie']
  pair_count=0;inversions=0;ties_skipped=0;near_tie_pairs=0
  for left,right in itertools.combinations(range(option_count),2):
   x=a['scores'][left]-a['scores'][right];y=b['scores'][left]-b['scores'][right]
   if (0<abs(x)<=1e-6) or (0<abs(y)<=1e-6):near_tie_pairs+=1
   if x==0 or y==0:ties_skipped+=1
   else:pair_count+=1;inversions+=int((x>0)!=(y>0))
  a['strict_pair_count']=pair_count;a['strict_inversions']=inversions;a['strict_inversion_fraction']=None if pair_count==0 else inversions/pair_count;a['pairs_with_exact_tie']=ties_skipped;a['near_tie_pairs_at_either_depth_1e-6']=near_tie_pairs
  group='options_'+str(option_count)+'|top_'+('stable_correct' if a['prediction']==b['prediction'] and a['correct'] else 'stable_wrong' if a['prediction']==b['prediction'] else 'changed')
  rank_strata[f][group]['n']+=1;rank_strata[f][group]['rank_changed']+=a['rank']!=b['rank'];rank_strata[f][group]['gold_rank_improved']+=b['gold_rank']<a['gold_rank'];rank_strata[f][group]['strict_inversions']+=inversions;rank_strata[f][group]['strict_pairs']+=pair_count;rank_strata[f][group]['pairs_with_ties']+=ties_skipped;rank_strata[f][group]['near_tie_pairs']+=near_tie_pairs
  margin=a['gold_margin'];bucket='<=-2' if margin<=-2 else '(-2,0]' if margin<=0 else '(0,2]' if margin<=2 else '>2'
  for group in ('options_'+str(option_count),'K1_gold_margin_'+bucket):strata[f][group]['n']+=1;strata[f][group][kinds[f]]+=1
 if kinds['joint']=='wrong->correct' and kinds['cyclic_mean']=='correct->wrong' or kinds['joint']=='correct->wrong' and kinds['cyclic_mean']=='wrong->correct':case_ids['rescue_damage_conflict'].append(ident)
 if kinds['joint']=='correct->wrong' and kinds['cyclic_mean']=='correct->wrong':case_ids['both_damage'].append(ident)
 matrix[kinds['joint']+'|'+kinds['cyclic_mean']]+=1
 changej=table[ident,'joint',1]['prediction']!=table[ident,'joint',4]['prediction'];changec=table[ident,'cyclic_mean',1]['prediction']!=table[ident,'cyclic_mean',4]['prediction']
 if changej and changec:
  switches['both_changed']+=1;switches['both_changed_same_final']+=same4;switches['both_changed_different_final']+=not same4
  if not same4:case_ids['both_changed_different_final'].append(ident)
  if same4 and not a4['correct']:case_ids['both_changed_same_wrong'].append(ident)
 elif changej:switches['joint_only_changed']+=1
 elif changec:switches['cyclic_only_changed']+=1
 else:switches['neither_changed']+=1
 items.append({'id':ident,'original_labels':docs[raw]['choices']['label'],'option_count':option_count,'endpoints':endpoints,'joint_transition':kinds['joint'],'cyclic_transition':kinds['cyclic_mean'],'joint_cyclic_same_K1':same1,'joint_cyclic_same_K4':same4,'joint_changed':changej,'cyclic_changed':changec,'leave_original_out':{str(k):leave_out[(ident,k)] for k in (1,4)}})
out=P.parent/'results';src_sha=hashlib.sha256(source.read_bytes()).hexdigest();data_sha=hashlib.sha256(data.read_bytes()).hexdigest();report={'scope':'historically exposed ARC-Challenge test; descriptive endpoint observations only','source':str(source.relative_to(R)),'source_sha256':src_sha,'raw_score_sha256':hashlib.sha256(source_raw).hexdigest(),'archive_sha256':hashlib.sha256(bundle).hexdigest(),'dataset_sha256':data_sha,'script_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'demonstrations_sha256':hashlib.sha256(demo_path.read_bytes()).hexdigest(),'n_questions':len(ids),'rows':len(table),'cyclic_definition':'arithmetic mean of mapped-back log-scores across cyclic rotations, including original order','limitations':['pooled cyclic is not a single permutation, majority vote, or independent evaluation','no K2/K3 observations; cannot localize transitions or exclude reversals','endpoint comparison does not establish robustness, label binding, or mechanism','scores and margin levels are analyzed within each view only'],'cross_readout_counts':{k:dict(sorted(v.items())) for k,v in counts.items()},'agreement_change_counts':dict(sorted(agreement.items())),'agreement_outcomes':dict(sorted(agreement_outcomes.items())),'case_ids':dict(case_ids),'case_intersections':{a+' & '+b:len(set(case_ids[a]) & set(case_ids[b])) for a,b in itertools.combinations(sorted(case_ids),2)},'leave_original_out_counts':{k:dict(v) for k,v in leave_counts.items()},'leave_original_out_provenance':'all rotations request SHA recreated; raw original component and pooled mean exact; direct non-original mean exactly equals (n*pooled-original)/(n-1)','cross_transition_counts':dict(sorted(matrix.items())),'prediction_switch_counts':dict(sorted(switches.items())),'near_tie_definition':'0 < absolute score difference <= 1e-6 at either observed depth; diagnostic only', 'within_protocol_rank_counts':{k:dict(v) for k,v in families.items()},'within_protocol_rank_strata':{k:{s:dict(c) for s,c in v.items()} for k,v in rank_strata.items()},'within_protocol_strata':{k:{s:dict(c) for s,c in v.items()} for k,v in strata.items()}}
(out/'HISTORICAL_ENDPOINT_OBSERVATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
with (out/'HISTORICAL_ENDPOINT_OBSERVATION.jsonl').open('w',encoding='utf8') as f:
 for item in items:f.write(json.dumps(item,ensure_ascii=False)+'\n')
print(json.dumps(report,ensure_ascii=False))
