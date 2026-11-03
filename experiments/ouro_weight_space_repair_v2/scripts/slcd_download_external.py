"""Local network fallback: pinned official parquet bytes, no evaluation."""
import hashlib,json,shutil,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
arc=json.loads((ROOT/'results/slcd_ARC_TEST_COMPLETE.json').read_text())
assert arc['status']=='complete' and arc['n']==1172 and arc['primary_correct']>717 and arc['other_benchmarks_authorized']
for task,repo,candidates in [('hellaswag','Rowan/hellaswag',['data/validation-00000-of-00001.parquet']),('openbookqa','allenai/openbookqa',['main/test-00000-of-00001.parquet','main/test/0000.parquet','main/openbookqa-test.parquet'])]:
    p=ROOT/f'data/slcd_{task}_raw.parquet';receipt=p.with_name(p.stem+'_SOURCE.json')
    if receipt.exists():
        source=json.loads(receipt.read_text());assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256'];continue
    assert not p.exists()
    info=json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/'+repo,timeout=60));available={r['rfilename'] for r in info['siblings']};relative=next(n for n in candidates if n in available)
    url='https://huggingface.co/datasets/'+repo+'/resolve/'+info['sha']+'/'+relative
    tmp=p.with_suffix('.download');assert not tmp.exists()
    with urllib.request.urlopen(url,timeout=120) as src,tmp.open('xb') as out:shutil.copyfileobj(src,out)
    tmp.replace(p);source=dict(repository=repo,revision=info['sha'],path=relative,url=url,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    receipt.write_text(json.dumps(source,indent=2),encoding='utf8')
    print(json.dumps(dict(task=task,bytes=p.stat().st_size,**source)),flush=True)
