"""Durable sequential train-fit-validation chain with a global three-hour cap."""
from pathlib import Path
import sys,subprocess,time,os,json
ROOT=Path(__file__).resolve().parents[1]
CURRENT_CHILD=None
CURRENT_START=None
def save(name,x):
    p=ROOT/'results'/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2),encoding='utf8');tmp.replace(p)
def main():
    global CURRENT_CHILD,CURRENT_START
    start=time.time();deadline=start+10800;runs=[]
    assert not (ROOT/'results/RUN_STATE.json').exists(),'Inspect existing attempt; no automatic retry'
    save('RUN_STATE.json',dict(status='starting',parent_pid=os.getpid(),start_unix=start,deadline=deadline))
    phases=[('head_train',['collect.py','head_train'],8500,True),('fit',['fit_heads.py'],120,False),('validation',['collect.py','validation'],10800,True),('analyze',['analyze.py'],180,False)]
    for phase,args,cap,gpu in phases:
        remaining=deadline-time.time()-15
        if remaining<60:raise RuntimeError('Global budget reserve')
        phase_start=time.time();env=dict(os.environ,CALIBRATION_DEADLINE=str(deadline-15))
        with (ROOT/'results'/f'{phase}.out').open('xb') as out,(ROOT/'results'/f'{phase}.err').open('xb') as err:
            p=subprocess.Popen([sys.executable,'-X','utf8','-X','faulthandler','-u',str(ROOT/'scripts'/args[0]),*args[1:]],stdout=out,stderr=err,env=env)
            CURRENT_CHILD=p;CURRENT_START=phase_start
            save('RUN_STATE.json',dict(status='running',phase=phase,pid=p.pid,parent_pid=os.getpid(),start_unix=start,phase_start_unix=phase_start,deadline=deadline,runs=runs))
            try:code=p.wait(timeout=min(cap,remaining))
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill.exe','/PID',str(p.pid),'/T','/F'],check=True,stdout=err,stderr=err);code=p.wait()
        rec=dict(phase=phase,pid=p.pid,exit_code=code,start_unix=phase_start,end_unix=time.time(),worker_seconds=time.time()-phase_start,gpu_reserved=gpu)
        CURRENT_CHILD=None
        runs.append(rec);save(phase+'_EXIT.json',rec);save('BUDGET_RUNNING.json',dict(runs=runs,gpu_charge_seconds=sum(r['worker_seconds'] for r in runs if r['gpu_reserved']),chain_wall_seconds=time.time()-start,allowance_seconds=10800))
        if code:
            save('RUN_STATE.json',dict(status='failed',phase=phase,start_unix=start,end_unix=time.time(),runs=runs));print(json.dumps(rec),flush=True);return
    save('RUN_STATE.json',dict(status='complete',start_unix=start,end_unix=time.time(),runs=runs));print(json.dumps(dict(status='complete',runs=runs)),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        if CURRENT_CHILD is not None and CURRENT_CHILD.poll() is None:
            subprocess.run(['taskkill.exe','/PID',str(CURRENT_CHILD.pid),'/T','/F'],check=True,capture_output=True);CURRENT_CHILD.wait()
        (ROOT/'results/SUPERVISOR_FATAL.json').write_text(json.dumps(dict(error=repr(exc),finished_unix=time.time(),active_interval_seconds=time.time()-CURRENT_START if CURRENT_START else 0)),encoding='utf8')
        raise
