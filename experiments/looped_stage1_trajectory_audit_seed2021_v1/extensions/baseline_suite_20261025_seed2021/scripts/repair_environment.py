import json
import time
from prepare_suite import ROOT,install
try:
    result=install();result['completed_unix']=time.time()
    (ROOT/'logs/environment_repair_status.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
except Exception as exc:
    (ROOT/'logs/environment_repair_status.json').write_text(json.dumps(dict(passed=False,error=repr(exc),completed_unix=time.time())),encoding='utf-8')
    raise
