"""CPU-only verification of every frozen E6-A decision, including LoopCD controls."""
import argparse,json,gzip,hashlib,sys
from pathlib import Path
import numpy as np
from E6A_METHODS import infer
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 p=Path(p)
 with (gzip.open(p,'rt',encoding='utf8') if p.suffix=='.gz' else p.open(encoding='utf8')) as f:
  for l in f:yield json.loads(l)
def main():
 p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.release);checked={}
 for folder,protocol,aggregation in [('joint5090','joint','scores'),('fas_char5090','full_answer','scores'),('fas_sum5090','full_answer','raw_sum'),('fas_token5090','full_answer','mean_token')]:
  r=root/'E6A'/folder;spec=read(r/'E6A_FROZEN_METHOD_SPEC.json');assert spec['aggregation']==aggregation
  source=root/'shared_capture'/(protocol+'_scores_features.jsonl.gz');caps={q['split']+':'+q['id']:q for q in lines(source)}
  result=r/'E6A_ALL_RESULTS.jsonl.gz';result=result if result.exists() else result.with_suffix('');n=0
  for row in lines(result):
   q=caps[row['split']+':'+row['id']];scores=np.array([k[aggregation] for k in q['native']]);name=row['method']
   if name in spec['methods']:
    actual=infer(scores,spec['methods'][name]);pred=actual['prediction'];depth=actual['selected_depth'];loops=actual['executed_loops']
   else:
    pred=int(np.argmax(q['replica_baselines'][name][aggregation]));depth=loops=4
   assert int(pred)==row['prediction'] and int(depth)==row['selected_depth'] and int(loops)==row['executed_loops'],(folder,row['id'],name)
   n+=1
  checked[folder]={'decisions':n,'all_predictions_depths_costs_equal':True,'frozen_spec_sha256':hashlib.sha256((r/'E6A_FROZEN_METHOD_SPEC.json').read_bytes()).hexdigest()}
 Path(a.out).write_text(json.dumps({'status':'PASS','new_fit':False,'gpu_forwards':0,'checked':checked},indent=2));print('E6A_RELEASE_REPLAY_PASS',checked)
if __name__=='__main__':main()
