"""Read-only project job monitor: snapshot every 600 s, never starts or restarts tasks."""
import os,time,json,subprocess,datetime,pathlib,re,shutil,psutil
BASE=pathlib.Path('/home/luansiyu/loop-latent/research_plans/natural_qkv_eight_corner_20261112');OUT=BASE/'monitor_10min.jsonl';INTERVAL=600
REGISTRY=BASE/'monitor_units.json'
def cmd(args):
 p=subprocess.run(args,text=True,capture_output=True,timeout=15);return p.returncode,p.stdout,p.stderr
def snapshot():
 record={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'interval_seconds':INTERVAL,'monitor_pid':os.getpid(),'units':{}}
 units=json.loads(REGISTRY.read_text());record['registry_sha256']=__import__('hashlib').sha256(REGISTRY.read_bytes()).hexdigest()
 for unit,expected_script in units.items():
  code,out,err=cmd(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','MainPID','-p','Result','-p','ExecMainStatus']);entry=dict(line.split('=',1) for line in out.splitlines() if '=' in line);pid=int(entry.get('MainPID','0'));entry['systemctl_exit']=code;entry['pid_exists']=pid>0 and pathlib.Path('/proc',str(pid)).exists();entry['expected_script_match']=expected_script.encode() in pathlib.Path('/proc',str(pid),'cmdline').read_bytes() if entry['pid_exists'] else False;record['units'][unit]=entry
  code,logs,err=cmd(['journalctl','--user','-u',unit,'-n','30','--no-pager']);lines=logs.splitlines();entry['recent_error_lines']=[l[-500:] for l in lines if re.search(r'Traceback|(?:^|\s)(?:Error|Exception|FAILED|failed|AssertionError|RuntimeError|ValueError|OutOfMemoryError)',l)];completed=[re.search(r'TRAIN_QUESTION_COMPLETE\s+(\d+)',l) for l in lines];entry['last_completed_number']=next((int(m.group(1)) for m in reversed(completed) if m),None)
 train=BASE/'train';record['train_complete_pairs']=sum(1 for p in train.glob('*.safetensors') if p.with_suffix('.json').is_file()) if train.exists() else 0;record['train_expected']=1119
 code,out,err=cmd(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']);record['gpu_query_exit']=code;record['gpu_csv']=out.strip();code,out,err=cmd(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits']);record['gpu_processes_csv']=out.strip().splitlines() if code==0 else [];record['ram_available_bytes']=psutil.virtual_memory().available;record['disk_free_bytes']=shutil.disk_usage(BASE).free;return record
next_tick=time.monotonic()
while True:
 try:r=snapshot()
 except BaseException as e:r={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'interval_seconds':INTERVAL,'monitor_pid':os.getpid(),'monitor_error_type':type(e).__name__}
 with OUT.open('a') as f:json.dump(r,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 print('MONITOR_SNAPSHOT',r['timestamp_utc'],'pairs',r.get('train_complete_pairs'),'error',r.get('monitor_error_type'),flush=True)
 next_tick+=INTERVAL;time.sleep(max(0,next_tick-time.monotonic()))
