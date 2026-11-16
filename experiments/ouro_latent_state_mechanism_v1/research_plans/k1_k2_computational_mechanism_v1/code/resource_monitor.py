import subprocess,json,time,pathlib,datetime
R=pathlib.Path('/root/ouro-computation-sol-20261116')
with (R/'RESOURCE_SAMPLES.jsonl').open('a') as f:
 while not ((R/'A_validation_COMPLETE.json').exists() and (R/'B_heldout_COMPLETE.json').exists()):
  v=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
  f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'uuid':v[0],'util_percent':float(v[1]),'memory_MiB':float(v[2]),'power_W':float(v[3]),'temperature_C':float(v[4])})+'\n');f.flush();time.sleep(5)
