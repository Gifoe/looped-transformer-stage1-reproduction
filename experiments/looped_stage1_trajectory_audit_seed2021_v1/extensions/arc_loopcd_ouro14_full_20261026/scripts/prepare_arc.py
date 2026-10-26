"""Freeze all1172 ARC-C requests; no subset, tuning or test exclusions."""
import hashlib
import importlib.metadata
import json
import sys
from windows_task_affinity import set_self
set_self('0xffff0000')
from common import ROOT, CFG, save
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_data import table, record, requests_for, write_manifest, snapshot_records

assert importlib.metadata.version('lm_eval') == CFG['lm_eval_version'] == '0.4.9'
train, revision = table('allenai/ai2_arc','ARC-Challenge','train')
test, _ = table('allenai/ai2_arc','ARC-Challenge','test')
validation, _ = table('allenai/ai2_arc','ARC-Challenge','validation')
assert len(test) == CFG['arc_count'] == 1172
rows = [record('arc_challenge','test',i,doc,revision) for i,doc in enumerate(test)]
assert len({r['id'] for r in rows}) == 1172
requests = requests_for(rows,train,None,'test')
assert all(r['fewshot_count']==25 and r['primary_metric']=='acc_norm' for r in requests)
test_audit = write_manifest('arc_full_test',requests)
checks = requests_for([record('arc_challenge','validation',i,doc,revision) for i,doc in enumerate(validation[:3])],train,None,'validation')
validation_audit = write_manifest('arc_parity_validation',checks)
from transformers import AutoTokenizer
from common import BASE, encode_pair
tok = AutoTokenizer.from_pretrained(BASE/'models/Ouro-1.4B',local_files_only=True)
longest = 0
for row in requests+checks:
    for pair in row['request_pairs']:
        prefix, answer = encode_pair(tok,*pair)
        longest=max(longest,len(prefix)+len(answer))
save('data/FULL_MANIFEST_AUDIT.json',dict(passed=True,full_test=test_audit,parity_validation=validation_audit,
    counts={'arc_challenge':1172},shots=25,maximum_request_tokens=longest,truncation=False,
    protocol_sha256=hashlib.sha256((ROOT/'configs/protocol.json').read_bytes()).hexdigest(),
    dataset_revision=revision,snapshots=snapshot_records,harness_version='0.4.9',
    order='canonical complete official test split',prior_test_questions_included=True))
print(json.dumps(dict(passed=True,full_test=test_audit,maximum_request_tokens=longest)),flush=True)
