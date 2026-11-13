import json,hashlib,time,argparse,collections
from pathlib import Path
import numpy as np
from E6A_METHODS import infer,fixed_specs,calibration_grid
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,m):Path(p).write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf8')
def load():
 p=Path(r'D:/ouro-e4-e5-20261112/inputs');ins=[json.loads(s) for s in (p/'runner_inputs.jsonl').read_text(encoding='utf8').splitlines()];labs={(r['split'],r['id']):r for r in map(json.loads,(p/'all_labels.jsonl').read_text(encoding='utf8').splitlines())}
 rows=[];seen=set();hashes=collections.defaultdict(list)
 for q in ins:
  key=(q['split'],q['id']);assert key not in seen;seen.add(key);l=labs[key];x=np.array(q['historical_scores']);assert x.shape==(4,len(q['targets'])) and 3<=x.shape[1]<=5 and np.isfinite(x).all();assert x.argmax(axis=1).tolist()==l['historical_predictions'];assert q['metadata_sha256']==l['metadata_sha256'];assert hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256']
  rows.append({**q,'gold':l['gold'],'demo':l['is_demonstration']});hashes[q['input_ids_sha256']].append(key)
 assert collections.Counter(q['split'] for q in rows)=={'train':1119,'validation':299,'test':1172};assert sum(q['demo'] for q in rows)==25
 return rows,{'n':2590,'candidate_counts':dict(collections.Counter(len(q['targets']) for q in rows)),'hardware':'historical RTX A6000 only','scores':'full candidate FP32 log-softmax vectors recovered from original metadata, not predictions','input_file_sha256':sha(p/'runner_inputs.jsonl'),'labels_file_sha256':sha(p/'all_labels.jsonl'),'exact_duplicate_input_groups':[v for v in hashes.values() if len(v)>1],'score_alignment':'same candidate target IDs/order and four native loops per question; variable candidate count retained','near_duplicates':'Pending normalized-text audit after tokenizer decode; no claim of semantic uniqueness.'}
def evaluate(rows,specs):
 allrows=[];summary=[]
 for name,spec in specs.items():
  correct=[];base=[];depth=[];notie=[]
  for q in rows:
   out=infer(q['historical_scores'],spec);pred=out['prediction'];b=int(np.argmax(q['historical_scores'][3]));gold=q['gold'];x=np.array(q['historical_scores']);nt=all(sum(z==z.max())==1 for z in x) and not out['top_tie'];correct.append(pred==gold);base.append(b==gold);depth.append(out['executed_loops']);notie.append(nt)
   allrows.append({'split':q['split'],'id':q['id'],'method':name,'track':spec['track'],'gold_evaluation_only':gold,'native_K4_prediction':b,'native_K4_correct':b==gold,'correct':pred==gold,'all_native_and_method_no_tie':nt,**out})
  c=np.array(correct);b=np.array(base);delta=c.astype(float)-b.astype(float);rng=np.random.default_rng(20261113);ci=np.quantile(delta[rng.integers(0,len(c),(10000,len(c)))].mean(axis=1)*100,[.025,.975]);mask=np.array(notie)
  summary.append({'method':name,'track':spec['track'],'n':len(c),'correct':int(c.sum()),'accuracy':float(c.mean()),'pp_vs_K4':float(delta.mean()*100),'rescue':int((c&~b).sum()),'damage':int((~c&b).sum()),'net_gain':int(delta.sum()),'paired_bootstrap95_pp':ci.tolist(),'no_tie_n':int(mask.sum()),'no_tie_accuracy':float(c[mask].mean()) if mask.any() else None,'no_tie_pp_vs_K4':float(delta[mask].mean()*100) if mask.any() else None,'mean_loops':float(np.mean(depth)),'relative_forward_loop_count':float(np.mean(depth)/4),'latency':'offline prefix simulation, not measured online','macro_f1':'NOT APPLICABLE: candidate indices are question-local options, not shared semantic classes'})
 return summary,allrows
def main():
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['dev','test'],required=True);args=a.parse_args();start=time.time();rows,audit=load();specs={**fixed_specs(),**calibration_grid()}
 if args.phase=='dev':
  assert not (ROOT/'TEST_EVALUATION_RECEIPT.json').exists();write(ROOT/'E6A_DATA_AUDIT.json',audit);results={};allrows=[]
  for split in ['train','validation']:
   rr=[r for r in rows if r['split']==split and not r['demo']];s,ar=evaluate(rr,specs);results[split]=s;allrows+=ar
  cal=[r for r in results['validation'] if r['track']=='Cal'];chosen=sorted(cal,key=lambda x:(-x['correct'],x['mean_loops'],x['method']))[0]['method']
  freeze={'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'primary_metric':'Test accuracy paired versus same-device Native K4','primary_ZL':'weighted4','primary_ZL_selection':'fixed before development evaluation; no task labels used','primary_Cal':chosen,'Cal_selection':'highest Validation correct, then lower mean loops, then method name; Train/Val labels explicitly allowed','methods':specs,'source_sha256':sha(__file__),'methods_sha256':sha(ROOT/'E6A_METHODS.py'),'data_audit_sha256':sha(ROOT/'E6A_DATA_AUDIT.json'),'data_identity':audit,'test_policy':'One frozen batch all candidates; no Test-tuned rules. Historical Test already exposed. Full-answer transfer separate freeze/baseline.'}
  write(ROOT/'E6A_FROZEN_METHOD_SPEC.json',freeze);write(ROOT/'DEVELOPMENT_SUMMARY.json',results)
  (ROOT/'DEVELOPMENT_RESULTS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in allrows),encoding='utf8');print({'phase':'dev','primary_Cal':chosen,'freeze_sha256':sha(ROOT/'E6A_FROZEN_METHOD_SPEC.json')})
 else:
  f=ROOT/'E6A_FROZEN_METHOD_SPEC.json';frozen=__import__('json').loads(f.read_text());assert frozen['source_sha256']==sha(__file__) and frozen['methods_sha256']==sha(ROOT/'E6A_METHODS.py');assert not (ROOT/'TEST_EVALUATION_RECEIPT.json').exists(),'Never silently repeat frozen Test evaluation'
  test=[r for r in rows if r['split']=='test'];s,ar=evaluate(test,frozen['methods']);write(ROOT/'TEST_SUMMARY.json',s);(ROOT/'E6A_ALL_RESULTS.jsonl').write_text((ROOT/'DEVELOPMENT_RESULTS.jsonl').read_text()+''.join(json.dumps(r)+'\n' for r in ar),encoding='utf8');write(ROOT/'TEST_EVALUATION_RECEIPT.json',{'status':'COMPLETE','freeze_sha256':sha(f),'n':len(test),'candidate_methods':len(s),'wall_seconds':time.time()-start,'scope':'historical homogeneous A6000 Joint-Option scoring, no new GPU forwards; simultaneous exploratory comparisons, not multiplicity corrected'});print(json.dumps(s))
if __name__=='__main__':main()
