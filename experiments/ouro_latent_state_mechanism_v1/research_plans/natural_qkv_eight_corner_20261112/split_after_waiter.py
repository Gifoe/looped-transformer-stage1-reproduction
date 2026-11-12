"""One-way validation -> audit -> analysis -> test -> audit -> analysis; fail closed."""
import sys,time,subprocess,json,hashlib
from pathlib import Path
B=Path('/home/luansiyu/loop-latent/research_plans/natural_qkv_eight_corner_20261112'); H='59438c1694501f7bf5dba9d6547fe037258a9bf57bba58aa6ff3cbb21956a7d7'; R=Path('/home/luansiyu/loop-latent/research_plans'); PY=sys.executable

def check_unit(unit):
 for _ in range(2880):
  p=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','Result'],capture_output=True,text=True,check=True);s=dict(x.split('=',1) for x in p.stdout.splitlines() if '=' in x)
  if s.get('ActiveState') in ('inactive','failed'):
   if s.get('Result')!='success':raise RuntimeError(f'{unit} failed {s}')
   return
  time.sleep(30)
 raise TimeoutError(unit)
def run(*args):
 print('START',*args,flush=True);subprocess.run(args,check=True)
print('WAIT_VALIDATION',flush=True);check_unit('ouro-qkv-eight-validation-20261112.service')
run(PY,'-u',str(B/'split_independent_audit.py'),'validation',H)
run(PY,'-u',str(B/'split_factorial_analysis.py'),'validation')
registry=B/'monitor_units.json';units=json.loads(registry.read_text());units['ouro-qkv-eight-test-20261112.service']='split_eight.py';tmp=registry.with_suffix('.tmp');tmp.write_text(json.dumps(units));tmp.replace(registry)
run('systemd-run','--user','--unit=ouro-qkv-eight-test-20261112','--property=KillMode=control-group','--setenv=PYTHONPATH='+':'.join(str(R/p) for p in ['natural_qkv_probe_code_20261112','natural_l18_probe_code_20261111','natural_branch_full_code_v5_20261111']),PY,'-u',str(B/'split_eight.py'),'test',H)
print('WAIT_TEST',flush=True);check_unit('ouro-qkv-eight-test-20261112.service')
run(PY,'-u',str(B/'split_independent_audit.py'),'test',H)
run(PY,'-u',str(B/'split_factorial_analysis.py'),'test')
print('VALIDATION_TEST_AUDITED_AND_ANALYZED',flush=True)
