"""CPU-only watcher safety tests; no monitor or GPU task started."""
import copy,fcntl,json,os,subprocess,sys,tempfile
from pathlib import Path
import pyarrow.parquet as pq
import ouro_watch as w

def check_reject(fn):
 try:fn()
 except RuntimeError:return
 raise AssertionError('unsafe case accepted')
def main():
 ok={'total_mib':49140,'used_mib':1000,'free_mib':48140,'util_percent':0,'compute_pids':[]}
 assert w.eligible(ok)
 for change in ({'free_mib':40959},{'free_mib':42000},{'util_percent':10},{'compute_pids':[123]}):
  x=dict(ok);x.update(change);assert not w.eligible(x)
 with tempfile.TemporaryDirectory(prefix='watch_test_',dir=w.ROOT) as temp:
  root=Path(temp);lock=root/'lock';fd=os.open(lock,os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  child=subprocess.run([sys.executable,'-c','import fcntl,sys; f=open(sys.argv[1],"r+");\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(0)\nsys.exit(3)',str(lock)]);assert child.returncode==0;os.close(fd)
  from cohort_runner import write_new_json
  exclusive=root/'exclusive.json';write_new_json(exclusive,{'initial':1});before=exclusive.read_bytes()
  try:write_new_json(exclusive,{'replacement':2})
  except FileExistsError:pass
  else:raise AssertionError('existing output overwritten')
  assert exclusive.read_bytes()==before
  oldroot,oldresults=w.ROOT,w.RESULTS;w.ROOT=root;w.RESULTS=root/'results';w.RESULTS.mkdir();(root/'shards').mkdir();check_reject(w.fresh_outputs);(root/'shards').rmdir()
  rows=[{'id':f'id{n}','choices':{'text':['x']*n}} for n in (3,4,5)];original=pq.read_table;pq.read_table=lambda path:type('T',(),{'to_pylist':lambda self:rows})()
  w.bound_hashes=lambda split:{'bound':'fake'};w.split_data_path=lambda split:root/'data.parquet';span={'records':[{'id':r['id']} for r in rows]};records=[]
  for row in rows:
   for j in range(len(row['choices']['text'])):
    records.append({'id':row['id'],'candidate':j,'evaluation_split':'validation','target_split':'train','causal_suffix_max_error':0.0,'repeated_max_error':0.0,'hook_max_error':0.0,'identity_max_error':0.0,'loops':[{'k':k,'historical':-1.,'capture':-1.,'absolute_error':0.,'depth_max_error':0.} for k in range(1,5)]})
  evidence=root/'evidence.json';evidence.write_text(json.dumps({'records':records,'max_allowed_error':0.0}));gate={'status':'passed','sha256':{'bound':'fake'},'producer_sha256':w.sha(w.R/'gpu_parity_gate.py'),'evidence_path':'evidence.json','evidence_sha256':w.sha(evidence),'checks':{k:'passed' for k in ('historical_scorer','independent_depths','identity_hook','bf16_roundtrip','repeatability','single_sequence','causal_mask')}};(w.RESULTS/'GPU_GATE_train.json').write_text(json.dumps(gate));assert w.gate_audit('train',span,w.RESULTS)['records']==12
  bad=copy.deepcopy(records);bad.pop();evidence.write_text(json.dumps({'records':bad,'max_allowed_error':0.0}));gate['evidence_sha256']=w.sha(evidence);(w.RESULTS/'GPU_GATE_train.json').write_text(json.dumps(gate));check_reject(lambda:w.gate_audit('train',span,w.RESULTS))
  bad=copy.deepcopy(records);bad[0]['identity_max_error']=0.01;evidence.write_text(json.dumps({'records':bad,'max_allowed_error':0.0}));gate['evidence_sha256']=w.sha(evidence);(w.RESULTS/'GPU_GATE_train.json').write_text(json.dumps(gate));check_reject(lambda:w.gate_audit('train',span,w.RESULTS))
  pq.read_table=original;w.ROOT,w.RESULTS=oldroot,oldresults
 print('PASS GPU thresholds, single-instance flock, existing-output refusal, evidence completeness and nonzero-error rejection')
if __name__=='__main__':main()
