import pathlib,subprocess,json,time,os,datetime
from freeze_guard import verify,exact_ids
R=pathlib.Path('/root/ouro-semantic-read-sol-20261117');P='/root/autodl-tmp/ouro-e4-e5/venv/bin/python';identity=verify();oldworker=17368
while pathlib.Path('/proc/'+str(oldworker)).exists():
 try:
  cmd=pathlib.Path('/proc/'+str(oldworker)+'/cmdline').read_bytes();stat=pathlib.Path('/proc/'+str(oldworker)+'/stat').read_text().split()[2]
 except FileNotFoundError:break
 if b'run_science.py' not in cmd or stat=='Z':break
 time.sleep(5)
assert json.loads((R/'development_SCIENCE_COMPLETE.json').read_text())['freeze_sha256']==identity
subprocess.run([P,'-u',str(R/'code/analyze.py'),'development','science'],check=True,cwd=R)
if not json.loads((R/'development_SEMANTIC_SUMMARY.json').read_text())['graph_directed_pairmessage_gate']:
 (R/'PROSPECTIVE_STOP.json').write_text(json.dumps({'status':'STOP_DEVELOPMENT_SEMANTIC_GATE','freeze_sha256':identity,'confirmation_neural_worlds':0,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));print('STOP_DEVELOPMENT_SEMANTIC_GATE',flush=True);raise SystemExit(0)
logs=R/'parallel_logs';logs.mkdir(exist_ok=True)
for stage in ['native','science']:
 children=[]
 for rank in range(3):
  stream=(logs/f'{stage}_confirmation_rank{rank}.log').open('w');child=subprocess.Popen([P,'-u',str(R/'code/run_shard.py'),stage,'confirmation',str(rank),'3'],cwd=R,stdout=stream,stderr=subprocess.STDOUT);children.append((child,stream));print('CONFIRMATION_SHARD_LIVE',stage,rank,child.pid,flush=True)
 for child,stream in children:
  code=child.wait();stream.close();assert code==0,('shard failed',stage,child.pid,code)
 subprocess.run([P,'-u',str(R/'code/analyze.py'),'confirmation',stage],check=True,cwd=R)
 if stage=='native' and not json.loads((R/'confirmation_NATIVE_CAPABILITY.json').read_text())['pass']:
  (R/'PROSPECTIVE_STOP.json').write_text(json.dumps({'status':'STOP_CONFIRMATION_CAPABILITY','freeze_sha256':identity,'confirmation_native_worlds':2048,'confirmation_intervention_worlds':0},indent=2));raise SystemExit(0)
 (R/('confirmation_'+stage.upper()+'_COMPLETE.json')).write_text(json.dumps({'status':'COMPLETE','freeze_sha256':identity,'worlds':2048,'groups':64,'parallel_workers':3,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
print('FROZEN_STUDY_ALL_PHASES_COMPLETE',flush=True)
