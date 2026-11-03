"""Render the remaining517 eligible TRAIN examples; no held-out data access."""
import hashlib,json
import pyarrow.parquet as pq
from runtime import ROOT
from build_train_manifests import fingerprint,record,render


def main():
    def rows(name):return [json.loads(l) for l in (ROOT/'data'/name).read_text().splitlines()]
    used=rows('slcd_discovery.jsonl')+rows('slcd_selection.jsonl')
    demos=json.loads((ROOT/'data/slcd_fixed_demonstrations.json').read_text())['documents']
    seen={fingerprint(d['question']) for d in demos};pool=[]
    for i,doc in enumerate(pq.read_table(ROOT/'data/raw/train.parquet').to_pylist()):
        key=fingerprint(doc['question'])
        if key in seen:continue
        seen.add(key);pool.append(record(doc,i))
    occupied={r['id'] for r in used}
    extra=[r for r in pool if r['id'] not in occupied]
    extra.sort(key=lambda r:hashlib.sha256(('slcd-confirmation|2021|'+r['id']).encode()).hexdigest())
    assert len(extra)==517 and len(pool)==1093
    data=render(extra,demos)
    assert not ({fingerprint(r['question']) for r in data}&{fingerprint(r['question']) for r in used})
    p=ROOT/'data/slcd_confirmation.jsonl';assert not p.exists()
    with p.open('x',encoding='utf8') as f:
        for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    receipt=dict(status='complete',n=517,scope='previously_unused_official_train_only',
        independent_of_old_discovery_and_selection=True,used_for_further_development_not_blind_test=True,
        data_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),fixed_demo_sha256=hashlib.sha256((ROOT/'data/slcd_fixed_demonstrations.json').read_bytes()).hexdigest(),
        ids=[r['id'] for r in data])
    (ROOT/'configs/slcd_CONFIRMATION_DATA_LOCK.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print('TRAIN_CONFIRMATION_RENDERED',len(data),receipt['data_sha256'])


if __name__=='__main__':main()
