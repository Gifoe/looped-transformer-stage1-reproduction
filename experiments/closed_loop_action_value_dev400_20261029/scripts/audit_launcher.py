import subprocess,sys,time,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];start=time.time()
with (root/'results/numeric_audit.out').open('xb') as out,(root/'results/numeric_audit.err').open('xb') as err:
    result=subprocess.run([sys.executable,'-X','utf8','-u',str(root/'scripts/audit_numeric.py')],stdout=out,stderr=err,timeout=120)
record=dict(exit_code=result.returncode,charged_seconds=time.time()-start)
(root/'results/NUMERIC_AUDIT_EXIT.json').write_text(json.dumps(record),encoding='utf8');print(json.dumps(record))
