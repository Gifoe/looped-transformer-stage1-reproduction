"""Recompute every registered train-only method from SHA-bound score records."""
import json
from slcd_core import ROOT,read,rows,sha,evaluate_registry,writecsv
registry=read('configs/slcd_METHODS.json')
assert sha('configs/slcd_METHODS.json')==read('configs/slcd_CONFIRMATION_METHOD_LOCK.json')['registry_sha256']
tokens={}
for split in ['discovery','selection']:
    result={}
    for job in ['initial','neighbors']:
        path=f'results/slcd_tokens_{split}_{job}.jsonl';receipt=read(path.replace('.jsonl','_COMPLETE.json'))
        assert receipt['status']=='complete' and sha(path)==receipt['score_file_sha256']
        for row in rows(path):
            if row['id'] not in result:result[row['id']]=row
            else:
                combined=result[row['id']]['methods']
                for key,value in row['methods'].items():
                    if key in combined:assert combined[key]==value
                    else:combined[key]=value
    tokens[split]=result
results=evaluate_registry(list(registry),registry,tokens)
writecsv('results/slcd_recomputed_development.csv',results)
print(json.dumps(dict(status='complete',methods=len(registry),rows=len(results),train_only=True)))
