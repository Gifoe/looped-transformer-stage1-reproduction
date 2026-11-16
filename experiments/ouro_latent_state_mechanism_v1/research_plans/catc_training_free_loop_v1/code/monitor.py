import pathlib,time,json,datetime,subprocess
R=pathlib.Path('/root/ouro-catc-sol-20261116');start=time.monotonic()
while not (R/'STATUS.json').exists():
 files=list((R/'results').glob('*.json'));err=[]
 for name in ['worker0.log','worker1.log','full_queue.log']:
  p=R/name
  if p.exists() and 'Traceback' in p.read_text():err.append(name)
 state={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'done':len(files),'total':1094,'elapsed_seconds':time.monotonic()-start,'worker_errors':err,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()}
 (R/'PROGRESS.json').write_text(json.dumps(state,indent=2))
 if err:break
 time.sleep(15)
