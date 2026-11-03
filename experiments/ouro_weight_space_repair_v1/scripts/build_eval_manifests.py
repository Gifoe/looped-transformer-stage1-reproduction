"""Open held-out ARC only after the corresponding immutable candidate lock."""
import argparse,hashlib,json
import pyarrow.parquet as pq
from runtime import ROOT,save
from build_train_manifests import render,write

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['validation','test'],required=True);a=ap.parse_args()
 lockname='CANDIDATE_LOCK.json' if a.split=='validation' else 'FINAL_LOCK.json'
 lockpath=ROOT/'configs'/lockname;lock=json.loads(lockpath.read_text())
 assert lock['candidate_definitions_frozen'] is True
 if a.split=='test':assert lock['test_evaluation_authorized'] is True
 demos=json.loads((ROOT/'data/fixed_demonstrations.json').read_text())['documents']
 rawpath=ROOT/'data/raw'/f'{a.split}.parquet';raw=pq.read_table(rawpath).to_pylist()
 assert len(raw)==(299 if a.split=='validation' else 1172)
 rows=[]
 for i,doc in enumerate(raw):
  rows.append(dict(id=f"arc_challenge/{a.split}/"+doc['id'],source_index=i,question=doc['question'],choices=doc['choices']['text'],gold=doc['choices']['label'].index(str(doc['answerKey'])),source_doc=doc))
 result=write(a.split+'.jsonl',render(rows,demos))
 audit=dict(status='passed',split=a.split,manifest=result,raw_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest(),lock_sha256=hashlib.sha256(lockpath.read_bytes()).hexdigest(),all_choice_counts_retained=True,demos=25)
 save(f'data/{a.split.upper()}_MANIFEST_AUDIT.json',audit);print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
