"""One sequential owner; independent long-running stages survive SSH detach."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parents[1]
CFG=json.loads((ROOT/'configs/protocol.json').read_text());PY=Path(CFG['environment_python'])
os.environ['PYTHONUTF8']='1';os.environ['HF_HOME']=str(ROOT/'cache/huggingface');os.environ['PIP_CACHE_DIR']=str(ROOT/'cache/pip')
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['PYTHONFAULTHANDLER']='1'

def status(**data):
    data.update(pid=os.getpid(),updated_unix=time.time())
    temp=ROOT/'STATUS.supervisor.tmp';temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    for i in range(30):
        try:temp.replace(ROOT/'STATUS.json');break
        except PermissionError:time.sleep(.05)
    print(json.dumps(data),flush=True)

def remaining():
    path=ROOT/'logs/gpu_budget.json'
    budget=json.loads((path if path.exists() else BASE/'logs/gpu_budget.json').read_text())
    return 50*3600-budget['used_seconds']

def run(tag,args,gpu=False):
    if gpu and remaining()<=120:raise RuntimeError('Cumulative GPU budget exhausted')
    status(stage=tag,state='running')
    start=time.monotonic()
    with (ROOT/'logs'/(tag+'.log')).open('a',encoding='utf-8') as stream:
        limit=remaining() if gpu else 7200
        process=subprocess.Popen([str(PY),'-u','-X','utf8',*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        try:code=process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            # Kill only the process tree that this supervisor just launched.
            # Windows venv redirectors can otherwise leave the interpreter alive.
            subprocess.run(['taskkill.exe','/PID',str(process.pid),'/T','/F'],stdout=stream,stderr=subprocess.STDOUT)
            if gpu:
                path=ROOT/'logs/gpu_budget.json'
                ledger=json.loads((path if path.exists() else BASE/'logs/gpu_budget.json').read_text())
                ledger['used_seconds']=50*3600
                ledger.setdefault('runs',[]).append(dict(tag=tag,state='supervisor_timeout_hard_cap',seconds=time.monotonic()-start))
                path.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
            raise
    if code:raise RuntimeError(f'{tag} failed ({code}); see logs/{tag}.log')
    status(stage=tag,state='passed',elapsed_s=time.monotonic()-start)

def resource_gate():
    resources=json.loads((ROOT/'logs/resource_status.json').read_text())
    repaired=ROOT/'logs/environment_repair_status.json'
    environment=resources['environment']
    if not environment.get('passed') and repaired.exists():environment=json.loads(repaired.read_text())
    assert resources['model']['passed'] and environment['passed']
    assert (ROOT/'data/manifest_audit.json').exists()
    result=dict(passed=True,model_revision=resources['model']['revision'],environment=environment,initial_install_error_retained='error' in resources['environment'])
    (ROOT/'logs/resource_gate.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['mc','math','latency','final'],required=True)
    parser.add_argument('--resume-math-validation',action='store_true');args=parser.parse_args()
    if args.resume_math_validation and args.phase!='math':parser.error('Validation recovery requires --phase math')
    resource_gate()
    if args.phase=='mc':
        run('protocol_tests',['-m','unittest','discover','-s','tests','-v'])
        for name in CFG['models']:run('parity_'+name,['scripts/verify_suite.py','--model',name],gpu=True)
        for name in CFG['models']:run('pilot_'+name,['scripts/score_mc.py','--model',name,'--split','pilot'],gpu=True)
        run('budget_estimate',['scripts/estimate_budget.py'])
        for name in CFG['models']:
            for split in ['validation','test']:run('mc_'+name+'_'+split,['scripts/score_mc.py','--model',name,'--split',split],gpu=True)
        run('mc_analysis',['scripts/analyze_suite.py','--stage','mc'])
        status(stage='mc_finished',state='partial',remaining=['mathematical generation','actual halting latency','final report'])
    elif args.phase=='math':
        recovery=None
        if args.resume_math_validation:
            recovery=json.loads((ROOT/'results/native_abort_validation_recovery_gate.json').read_text())
            assert recovery['passed'] and recovery['protocol_sha256']==hashlib.sha256((ROOT/'configs/protocol.json').read_bytes()).hexdigest()
            files=[ROOT/'scripts/common.py',ROOT/'scripts/methods.py',BASE/'scripts/qwen_recipe.py',ROOT/'scripts/generate_math.py']
            fingerprint=hashlib.sha256(''.join(p.name+hashlib.sha256(p.read_bytes()).hexdigest() for p in files).encode()).hexdigest()
            assert recovery['generation_implementation_sha256']==fingerprint
            for name,entry in recovery['validation_prefixes'].items():
                path=ROOT/f'trajectories/math_{name}_validation.jsonl'
                assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
                assert not (ROOT/f'trajectories/math_{name}_test.jsonl').exists()
            status(stage='validation_recovery_verified',state='running',preserved_records=recovery['validation_prefixes'])
        else:
            run('audit_tokenizer_export',['scripts/export_audit_tokenizers.py'])
            run('mc_diagnostics',['scripts/analyze_diagnostics.py'])
            for name in ['Ouro-1.4B','Ouro-2.6B']:run('generation_parity_'+name,['scripts/generate_math.py','--model',name,'--stage','parity'],gpu=True)
            for name in ['Ouro-1.4B','Ouro-2.6B']:run('generation_pilot_'+name,['scripts/generate_math.py','--model',name,'--stage','pilot'],gpu=True)
            run('generation_budget',['scripts/estimate_budget.py','--generation'])
        for name in ['Ouro-1.4B','Ouro-2.6B']:
            if recovery and recovery['validation_prefixes'][name]['complete_validation']:
                status(stage='generation_validation_'+name,state='passed',recovery_skip_already_complete=True)
            else:run('generation_validation_'+name,['scripts/generate_math.py','--model',name,'--stage','validation'],gpu=True)
            if recovery and name=='Ouro-1.4B':
                run('recovered_validation_integrity',['scripts/audit_math_prefix.py','--model',name,'--require-complete'])
                run('recovery_lineage',['scripts/verify_recovery_lineage.py'])
        run('generation_selection',['scripts/analyze_suite.py','--stage','select_math'])
        run('generation_selected_budget',['scripts/estimate_budget.py','--generation','--selected'])
        for name in ['Ouro-1.4B','Ouro-2.6B']:run('generation_test_'+name,['scripts/generate_math.py','--model',name,'--stage','test'],gpu=True)
        status(stage='math_finished',state='partial',remaining=['actual halting latency','final report'])
        for name in ['Ouro-1.4B','Ouro-2.6B']:run('actual_halting_'+name,['scripts/actual_halting.py','--model',name],gpu=True)
        run('actual_halting_analysis',['scripts/analyze_latency.py'])
        status(stage='experiments_finished',state='awaiting_research_review',remaining=['final CPU audit','evidence-based research decision','report and package'])
    elif args.phase=='latency':
        for name in ['Ouro-1.4B','Ouro-2.6B']:run('actual_halting_'+name,['scripts/actual_halting.py','--model',name],gpu=True)
        run('actual_halting_analysis',['scripts/analyze_latency.py'])
        status(stage='latency_finished',state='partial',remaining=['final report'])
    else:
        run('final_statistics',['scripts/analyze_suite.py','--stage','all'])
        run('final_diagnostics',['scripts/analyze_diagnostics.py'])
        run('final_mc_reference_comparison',['scripts/compare_mc_rebuild.py'])
        run('final_latency_statistics',['scripts/analyze_latency.py'])
        run('final_tests',['-m','unittest','discover','-s','tests','-v'])
        run('final_validation_trace_integrity',['scripts/audit_math_prefix.py','--model','all','--require-complete'])
        run('historical_window_audit',['scripts/audit_historical_windows.py'])
        run('completion_audit',['scripts/audit_suite.py'])
        run('report',['scripts/write_report.py'])
        status(stage='completed',state='completed_with_source_limitations')
        # Packaging must not mutate STATUS or its captured manifest afterwards.
        subprocess.check_call([str(PY),'-u','-X','utf8','scripts/package_suite.py'],cwd=ROOT)

if __name__=='__main__':
    try:main()
    except Exception as exc:status(stage='failed',state='failed',error=repr(exc));traceback.print_exc();raise
