from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import requests
import time
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
urls=[
 'https://hf-mirror.com/ByteDance/Ouro-1.4B/resolve/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/model.safetensors?download=true',
 'https://hf-mirror.com/ByteDance/Ouro-1.4B/resolve/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/model.safetensors?download=1',
 'https://pypi.tuna.tsinghua.edu.cn/simple/numpy/',
 'https://mirrors.aliyun.com/pypi/simple/numpy/',
 'https://modelscope.cn/api/v1/models/ByteDance/Ouro-1.4B/repo/files?Revision=master&Recursive=true',
]
def probe(url):
    start=time.monotonic()
    try:
        s=requests.Session();s.trust_env=False
        with s.get(url,headers={'Range':'bytes=0-1023'},stream=True,timeout=(8,12)) as r:
            info=dict(url=url,status=r.status_code,final_host=urlsplit(r.url).hostname,
                      content_length=r.headers.get('Content-Length'),content_range=r.headers.get('Content-Range'))
            first=next(r.iter_content(1024),b'');info.update(first_bytes=len(first),elapsed_s=time.monotonic()-start)
            if 'modelscope.cn' in url:info['text']=first.decode(errors='replace')[:500]
            return info
    except Exception as e:return dict(url=url,error=str(e),elapsed_s=time.monotonic()-start)
with ThreadPoolExecutor(5) as pool:results=list(pool.map(probe,urls))
(ROOT/'logs/network_diagnostic.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results),flush=True)
