"""Run passed stages sequentially. Fail closed on implementation/resource errors."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/'.venv/Scripts/python.exe'
os.environ['PYTHONUTF8']='1'
os.environ['HF_HOME']=str(ROOT/'cache/huggingface')
os.environ['PIP_CACHE_DIR']=str(ROOT/'cache/pip')
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM']='false'

def status(**data):
    data['updated_unix']=time.time()
    (ROOT/'STATUS.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(data),flush=True)

def run(tag,args,gpu=False):
    budget=ROOT/'logs/gpu_budget.json'
    used=json.loads(budget.read_text())['used_seconds'] if budget.exists() else 0
    remaining=50*3600-used
    if gpu and remaining<=120:raise RuntimeError('GPU budget exhausted')
    status(stage=tag,state='running')
    with (ROOT/'logs'/(tag+'.log')).open('a',encoding='utf-8') as log:
        r=subprocess.run([str(PY),'-u','-X','utf8',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=remaining if gpu else 3600)
    if r.returncode:raise RuntimeError(f'{tag} failed ({r.returncode}); see logs/{tag}.log')
    status(stage=tag,state='passed')

def main():
    status(stage='waiting_for_resources',state='running')
    deadline=time.monotonic()+1800
    while True:
        ready=ROOT/'logs/resource_status.json'
        if ready.exists():
            data=json.loads(ready.read_text())
            if data.get('environment',{}).get('smoke_returncode')!=0:raise RuntimeError('Environment smoke check has not passed')
            ouro=data.get('ouro',{})
            if 'error' in ouro or any('error' in f for f in ouro.get('files',[])):raise RuntimeError('Ouro resource acquisition failed')
            break
        if time.monotonic()>deadline:raise RuntimeError('Resources did not finish within 30 minutes')
        time.sleep(10)
    run('data_acquisition',['scripts/prepare_data.py'])
    run('protocol_tests',['-m','unittest','discover','-s','tests','-v'])
    run('ouro_parity',['scripts/ouro_audit.py','--stage','parity'],gpu=True)
    run('performance_preflight',['scripts/ouro_audit.py','--stage','benchmark'],gpu=True)
    run('ouro_pilot',['scripts/ouro_audit.py','--stage','pilot'],gpu=True)
    run('ouro_formal',['scripts/ouro_audit.py','--stage','formal'],gpu=True)
    run('ouro_analysis',['scripts/analyze.py'])
    status(stage='ouro_core_analysis_finished',state='partial',qwen='blocked: official source repository HTTP 404',math_generation='not yet executed')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status(stage='blocked',state='failed',error=str(exc));traceback.print_exc();sys.exit(1)
