"""Public-source revision/availability metadata only; no corpus download or GPU."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import urllib.error
import urllib.request
from linux_runtime import ROOT,save

inventory=json.loads((ROOT/'configs/REBUILD_INPUT_INVENTORY.json').read_text())
repos=sorted({r['repo'] for r in inventory['sources']})


def get(repo):
    url='https://huggingface.co/api/datasets/'+repo+'?expand[]=sha&expand[]=gated'
    try:
        with urllib.request.urlopen(url,timeout=20) as f:d=json.load(f)
        return repo,dict(status='metadata_available',revision=d.get('sha'),gated=d.get('gated'))
    except urllib.error.HTTPError as e:return repo,dict(status='http_error',http_status=e.code)
    except Exception as e:return repo,dict(status='metadata_failed',error_type=type(e).__name__)


out={}
with ThreadPoolExecutor(max_workers=4) as pool:
    for f in as_completed([pool.submit(get,r) for r in repos]):
        repo,value=f.result();out[repo]=value
        save(ROOT/'configs/coread_source_revisions.json',dict(sorted(out.items())))
print(json.dumps(dict(repos=len(out),available=sum(v['status']=='metadata_available' for v in out.values()),
    gated=sum(bool(v.get('gated')) for v in out.values()),failed=[r for r,v in out.items() if v['status']!='metadata_available'])))
