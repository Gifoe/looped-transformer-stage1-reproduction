"""Shared controller child lifecycle; callers retain lock around entire call."""
import time,subprocess
from ouro_watch import owned_process

def run_owned(args,log,cwd,env,monitor,period=5):
 with log.open('x') as f:
  with owned_process(args,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT) as proc:
   started=time.monotonic()
   while proc.poll() is None:
    time.sleep(period);monitor(proc,time.monotonic()-started)
   if proc.returncode:raise RuntimeError('child exited '+str(proc.returncode))

def stage_admissible(snapshot,required_gpu_bytes,gib):
 return snapshot['gpu']['free_mib']*1024**2>required_gpu_bytes and snapshot['disk_free_bytes']>=20*gib and snapshot['memory']['MemAvailable']>=8*gib
