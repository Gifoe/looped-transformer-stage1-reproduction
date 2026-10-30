"""Official build_bench ordering/filtering from hash-pinned raw parquet payloads.

Avoids failing Hub HEAD requests. No evaluation or training samples are redefined.
"""
import hashlib
import json
import random
import time
import urllib.request
from linux_runtime import ROOT,configure,save

configure('cpu')
import pyarrow.parquet as pq
raw=ROOT/'data/raw_bench';raw.mkdir(exist_ok=True)
specs=json.loads((ROOT/'configs/benchmark_payloads.json').read_text())
tables={}
for s in specs:
    p=raw/(s['name']+'.parquet')
    if not p.exists():
        for attempt in range(4):
            try:
                with urllib.request.urlopen(s['url'],timeout=40) as response:payload=response.read()
                assert len(payload)==s['bytes'] and hashlib.sha256(payload).hexdigest()==s['sha256']
                tmp=p.with_suffix('.download');tmp.write_bytes(payload);tmp.replace(p);break
            except Exception:
                if attempt==3:raise
                time.sleep(1)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==s['sha256']
    tables[s['name']]=pq.read_table(p).to_pylist()


def arc(name):
    rows=[]
    for r in tables[name]:
        labels=r['choices']['label']
        if len(labels)==4 and r['answerKey'] in labels:
            rows.append(dict(src='arc_easy' if 'easy' in name else 'arc_challenge',q=r['question'],choices=r['choices']['text'],answer=labels.index(r['answerKey'])))
    return rows


def mmlu(name):
    return [dict(src='mmlu',subject=r['subject'],q=r['question'],choices=r['choices'],answer=r['answer']) for r in tables[name] if len(r['choices'])==4]


rng=random.Random(0)
test=mmlu('mmlu_test');rng.shuffle(test)
val=mmlu('mmlu_validation');rng.shuffle(val)
av=arc('arc_challenge_validation');rng.shuffle(av)
B=dict(train=arc('arc_challenge_train')+arc('arc_easy_train')+test[2000:]+val[300:],
    val=av[:300]+val[:300],test=arc('arc_challenge_test')+test[:2000])
p=ROOT/'upstream/data/bench.json'
assert not p.exists(),'refuse to overwrite a previous benchmark'
with p.open('w') as f:json.dump(B,f)
sha=hashlib.sha256(p.read_bytes()).hexdigest()
scores=json.loads((ROOT/'sources/published/results/magic-nl23/bench.json').read_text())
audit=json.loads((ROOT/'sources/published/results/magic-nl28/bench_audited.json').read_text())
assert [dict(src=r['src'],answer=r['answer']) for r in B['test']]==scores['questions']
manifest=dict(data_sha256=sha,author_data_sha256=audit['data_sha256'],exact_author_bytes=sha==audit['data_sha256'],
    source_answer_metadata_matches=True,sizes={k:len(v) for k,v in B.items()},raw_payloads=specs,
    implementation='Same transformations, ordering, seed and JSON serializer as official build_bench.py; raw parquet supplied directly')
save(ROOT/'data/BENCH_MANIFEST.json',manifest)
print(json.dumps(manifest,indent=2))
