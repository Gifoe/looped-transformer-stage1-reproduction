"""One durable task owner with verified inherited affinity and failure charging."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time
from windows_task_affinity import set_self,get_mask

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/protocol.json').read_text())
SHARED=Path(CFG['shared_budget_root'])
OUT=ROOT/'logs/arc_controller'
PY=CFG['environment_python']


def put(path,value):
    temp=path.with_suffix(path.suffix+'.writing')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def live():
    p=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',
        "Get-CimInstance Win32_Process|Where-Object{$_.Name -eq 'python.exe' -and $_.CommandLine -match 'score_arc[.]py|generate_math[.]py|actual_halting[.]py|run_suite[.]py|score_mc[.]py|score_full[.]py'}|Select-Object ProcessId,ParentProcessId,CommandLine|ConvertTo-Json"],capture_output=True,text=True,check=True)
    return p.stdout.strip()


def main():
    assert not live(),'Another experiment owner is live'
    assert not OUT.exists(),'Controller already launched; inspect its actual process instead of repeating it'
    preparation=json.loads((ROOT/'data/FULL_MANIFEST_AUDIT.json').read_text())
    assert preparation['passed'] and preparation['counts']['arc_challenge']==1172
    assert preparation['protocol_sha256']==hashlib.sha256((ROOT/'configs/protocol.json').read_bytes()).hexdigest()
    OUT.mkdir()
    put(OUT/'AFFINITY.json',set_self('0xffff0000'))
    before=json.loads((SHARED/'logs/gpu_budget.json').read_text());remaining=50*3600-before['used_seconds']
    assert remaining>180 and abs(before['used_seconds']-before['stage1_used_seconds']-sum(r['seconds'] for r in before['runs']))<1e-4
    sources={}
    for name in ['common.py','methods.py','score_arc.py','arc_controller.py','analyze_arc.py','windows_task_affinity.py']:
        source=ROOT/'scripts'/name;shutil.copyfile(source,OUT/name)
        sources[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    put(OUT/'LOADED_SOURCES.json',sources)
    env=dict(os.environ,PYTHONUTF8='1',PYTHONFAULTHANDLER='1',HF_HOME=str(SHARED/'cache/huggingface'),
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    started=time.time();clock=time.monotonic();verified=False;timed_out=False;stop=False;affinity_error=None
    with (OUT/'worker_stdout.log').open('w',encoding='utf-8') as out,(OUT/'worker_stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen([PY,'-u','-X','utf8','-X','faulthandler',str(ROOT/'scripts/score_arc.py')],cwd=ROOT,env=env,stdout=out,stderr=err)
        put(OUT/'OWNER.json',dict(controller_pid=os.getpid(),child_redirector_pid=child.pid,started_unix=started))
        while child.poll() is None:
            if not verified:
                active=json.loads((SHARED/'logs/active_gpu.json').read_text())
                if active.get('tag')=='full_ARC_LoopCD_Ouro1.4B' and active.get('started_unix',0)>=started-1:
                    try:
                        mask=get_mask(active['pid'])[0]
                        if mask!=0xffff0000:affinity_error='Actual worker mask '+hex(mask)
                        else:
                            verified=True;put(OUT/'WORKER_AFFINITY.json',dict(pid=active['pid'],mask=hex(mask),verified=True))
                    except OSError:pass
            timed_out=time.monotonic()-clock>min(7200,remaining-120)
            stop=(OUT/'STOP_REQUEST.json').exists()
            if timed_out or stop or affinity_error:
                subprocess.run(['taskkill.exe','/PID',str(child.pid),'/T','/F'],stdout=err,stderr=err);child.wait();break
            time.sleep(.25)
        code=child.wait()
    assert not live(),'Descendant still live; do not charge or relaunch'
    elapsed=time.monotonic()-clock
    ledger=json.loads((SHARED/'logs/gpu_budget.json').read_text())
    new=ledger['runs'][len(before['runs']):]
    matches=[r for r in new if r['tag']=='full_ARC_LoopCD_Ouro1.4B']
    assert len(matches)<=1
    if not matches:
        active=json.loads((SHARED/'logs/active_gpu.json').read_text())
        pid=active['pid'] if active.get('tag')=='full_ARC_LoopCD_Ouro1.4B' and active.get('started_unix',0)>=started-1 else child.pid
        ledger['runs'].append(dict(tag='full_ARC_LoopCD_Ouro1.4B',pid=pid,seconds=elapsed,
            state='conservative_native_or_controlled_abort',child_exit_code=code,
            accounting='Entire controlled subprocess interval including imports/startup/termination; no prior closed run double-counted'))
        ledger['used_seconds']+=elapsed;ledger['extension_used_seconds']=ledger['used_seconds']-ledger['stage1_used_seconds']
        assert abs(ledger['used_seconds']-ledger['stage1_used_seconds']-sum(r['seconds'] for r in ledger['runs']))<1e-4
        put(SHARED/'logs/gpu_budget.json',ledger)
        put(SHARED/'logs/active_gpu.json',dict(pid=pid,tag='full_ARC_LoopCD_Ouro1.4B',state='failed',finished_unix=time.time()))
    receipt=dict(native_exit_code=code,timeout=timed_out,stop_requested=stop,worker_affinity_verified=verified,
                 affinity_error=affinity_error,elapsed_s=elapsed,charged_cumulative_gpu_hours=ledger['used_seconds']/3600)
    put(OUT/'GPU_EXIT.json',receipt)
    assert code==0 and verified and not timed_out and not stop and not affinity_error,receipt
    assert all(hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==value for name,value in sources.items())
    with (OUT/'analysis.log').open('w',encoding='utf-8') as stream:
        analysis=subprocess.run([PY,'-u','-X','utf8','-X','faulthandler',str(ROOT/'scripts/analyze_arc.py')],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    put(OUT/'CPU_ANALYSIS_EXIT.json',dict(exit_code=analysis.returncode))
    assert analysis.returncode==0,'Read retained CPU analysis log; GPU outputs must not be rerun'
    result=json.loads((ROOT/'results/ARC_REPLICATION_GATE.json').read_text())
    put(OUT/'DONE.json',dict(completed=True,gpu_native_exit_code=0,cpu_analysis_exit_code=0,
        replication_gate_passed=result['replication_gate_passed'],gsm8k_allowed=result['gsm8k_allowed']))
    print(json.dumps(dict(completed=True,replication_gate_passed=result['replication_gate_passed'])),flush=True)


if __name__=='__main__':main()
