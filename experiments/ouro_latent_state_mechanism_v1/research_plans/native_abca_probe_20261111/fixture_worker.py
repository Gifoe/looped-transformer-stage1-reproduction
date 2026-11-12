import os,time,subprocess,sys
import signal
def stop(sig,frame):
 time.sleep(2);raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(180)'])
with open(os.environ['FIXTURE_PIDS'],'w') as f:f.write(str(os.getpid())+' '+str(p.pid));f.flush();os.fsync(f.fileno())
time.sleep(180)
