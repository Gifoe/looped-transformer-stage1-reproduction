"""Draft attempt-local independent probe replay; injectable executor only for CPU tests."""
import json,sys,subprocess
from pathlib import Path
from pqr_integrity import allocate_attempt,binding_guard
if not __debug__:raise RuntimeError('optimized Python forbidden')
def recheck(out,binding_path,binding_sha,probe_path,shards,longest,coverage_script,sha,execute=subprocess.run):
 binding_guard(binding_path,binding_sha,sha);probe_path=Path(probe_path).resolve();probe_sha=sha(probe_path);pr=json.loads(probe_path.read_text());assert sha(probe_path)==probe_sha
 assert pr['status']=='PQR_LONGEST_PROBE_CUDA_REPLAY_ZERO' and pr['binding_sha256']==binding_sha and pr['split']==longest['split'] and pr['expected_questions']==pr['complete_questions']==1 and pr['replayed_condition_logits']==4 and len(pr['questions'])==1 and pr['questions'][0]['id']==longest['id']
 attempt,path=allocate_attempt(out.parent,out.name);execute([sys.executable,str(coverage_script),'--split',longest['split'],'--shards',str(shards),'--binding',str(binding_path),'--report',str(path),'--probe-id',longest['id']],check=True)
 rr=json.loads(path.read_text());assert rr['status']=='PQR_LONGEST_PROBE_CUDA_REPLAY_ZERO' and rr['questions']==pr['questions'] and rr['binding_sha256']==pr['binding_sha256'] and sha(probe_path)==probe_sha;binding_guard(binding_path,binding_sha,sha)
 return attempt,str(path),sha(path)
