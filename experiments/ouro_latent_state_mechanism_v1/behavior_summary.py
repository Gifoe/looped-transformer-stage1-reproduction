"""Compute observed loop trajectories; no latent mechanism claim."""
import argparse,collections,hashlib,json,math
from pathlib import Path
import pyarrow.parquet as pq
from ledger_guard import require_production_ledger
from cohort_runner import write_new_json
from cohort_runner import PROJECT,REPO,split_data_path,bound_hashes,sha,write_json

def summarize(records):
 if not records:return {'n':0,'correct_counts':[0]*4,'accuracy':[None]*4,'transitions':[],'trajectory_counts':{}}
 n=len(records);counts=[sum(int(r['predictions'][k]==r['gold']) for r in records) for k in range(4)];transitions=[];patterns=collections.Counter()
 for r in records:patterns[''.join('1' if p==r['gold'] else '0' for p in r['predictions'])]+=1
 for k in range(3):
  rescue=sum(r['predictions'][k]!=r['gold'] and r['predictions'][k+1]==r['gold'] for r in records);damage=sum(r['predictions'][k]==r['gold'] and r['predictions'][k+1]!=r['gold'] for r in records)
  transitions.append({'from':k+1,'to':k+2,'rescue':rescue,'damage':damage,'net':rescue-damage,'prediction_changes':sum(r['predictions'][k]!=r['predictions'][k+1] for r in records)})
  assert counts[k+1]-counts[k]==rescue-damage
 total_net=counts[3]-counts[0];first_net=counts[1]-counts[0]
 return {'n':n,'correct_counts':counts,'accuracy':[x/n for x in counts],'transitions':transitions,'trajectory_counts':dict(sorted(patterns.items())),'prediction_trajectory_counts':dict(collections.Counter(','.join(map(str,r['predictions'])) for r in records)),'top_score_ties':[sum(r.get('top_score_ties',[False]*4)[k] for r in records) for k in range(4)],'any_score_ties':[sum(r.get('any_score_ties',[False]*4)[k] for r in records) for k in range(4)],'total_net_gain':total_net,'first_transition_net_gain':first_net,'first_transition_share_of_total_net':None if total_net<=0 else first_net/total_net}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',required=True,choices=['train','validation','test']);ap.add_argument('--shards',required=True);ap.add_argument('--coverage',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 shards=Path(a.shards).resolve();shards.relative_to(PROJECT.resolve());out=Path(a.out).resolve();out.relative_to(PROJECT.resolve());cov=json.loads(Path(a.coverage).read_text());require_production_ledger(cov);counts=cov['counts']
 if counts['complete_questions']!=counts['expected_questions'] or counts['complete_requests']!=counts['expected_requests'] or counts['failed_questions'] or counts['missing_questions']:raise RuntimeError('coverage incomplete')
 rows=pq.read_table(split_data_path(a.split)).to_pylist()
 if len(rows)!=counts['expected_questions'] or len({r['id'] for r in rows})!=len(rows):raise RuntimeError('coverage denominator or IDs mismatch')
 coverage_rows=cov['questions'];ledger={x['id']:x for x in coverage_rows}
 if len(ledger)!=len(coverage_rows) or set(ledger)!={r['id'] for r in rows} or any(x['status']!='complete' or x['split']!=a.split for x in coverage_rows):raise RuntimeError('coverage ledger missing or inconsistent')
 expected=bound_hashes(a.split);demo_doc=json.loads((REPO/'experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json').read_text());demos=set(demo_doc['ids']);normalize=lambda q:' '.join(q.lower().split());demo_questions={normalize(doc['question']) for doc in demo_doc['documents']};items=[]
 for row in rows:
  name=hashlib.sha256(row['id'].encode()).hexdigest();meta=json.loads((shards/(name+'.json')).read_text());payload=shards/(name+'.safetensors')
  if meta['id']!=row['id'] or meta['split']!=a.split or meta['sha256']!=expected or not payload.is_file() or sha(payload)!=meta['tensor_sha256'] or ledger[row['id']]['tensor_sha256']!=meta['tensor_sha256'] or ledger[row['id']]['metadata_sha256']!=sha(shards/(name+'.json')):raise RuntimeError('stale or corrupt shard: '+row['id'])
  gold=row['choices']['label'].index(row['answerKey']);records=meta['records'];n=len(row['choices']['text'])
  if len(records)!=n or [r['candidate'] for r in records]!=list(range(n)) or any(len(r['scores'])!=4 or not all(math.isfinite(z) for z in r['scores']) for r in records):raise RuntimeError('invalid candidate coverage or scores')
  scores=[r['scores'] for r in records];pred=[max(range(len(scores)),key=lambda j:scores[j][k]) for k in range(4)]
  if pred!=meta['predictions'] or gold!=meta['gold_index']:raise RuntimeError('outcome mismatch')
  top_ties=[sum(scores[j][k]==max(scores[h][k] for h in range(len(scores))) for j in range(len(scores)))>1 for k in range(4)]
  any_ties=[len({scores[j][k] for j in range(len(scores))})<len(scores) for k in range(4)]
  items.append({'id':row['id'],'gold':gold,'predictions':pred,'top_score_ties':top_ties,'any_score_ties':any_ties,'is_demonstration':row['id'] in demos,'demonstration_question_text_overlap':normalize(row['question']) in demo_questions})
 report={'scope':'observational behavior; no causal claim','split':a.split,'independence_note':'test historically exposed; fixed demonstration overlap is not independent','all':summarize(items),'demonstration_overlap':summarize([x for x in items if x['is_demonstration']]),'non_demonstration':summarize([x for x in items if not x['is_demonstration']]),'demonstration_question_text_overlap':summarize([x for x in items if x['demonstration_question_text_overlap']]),'no_demonstration_question_text_overlap':summarize([x for x in items if not x['demonstration_question_text_overlap']]),'question_trajectories':items};write_new_json(out,report)
if __name__=='__main__':main()
