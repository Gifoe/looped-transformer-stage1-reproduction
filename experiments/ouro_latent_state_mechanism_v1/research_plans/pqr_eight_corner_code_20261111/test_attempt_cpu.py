import json,hashlib,tempfile,pathlib
from pqr_attempt import recheck
from pqr_integrity import validate_attempt,binding_guard,require_fresh_probe
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as tmp:
 root=pathlib.Path(tmp);binding=root/'binding.json';binding.write_text(json.dumps({'binding':{'x':1},'longest_request':{'id':'x','split':'test'}}));bsha=sha(binding);longest={'id':'x','split':'test'};probe=root/'probe.json';r={'status':'PQR_LONGEST_PROBE_CUDA_REPLAY_ZERO','binding_sha256':bsha,'split':'test','expected_questions':1,'complete_questions':1,'replayed_condition_logits':4,'questions':[{'id':'x','metadata_sha256':'m','logits_sha256':'t'}]};probe.write_text(json.dumps(r));out=root/'same-output';out.mkdir()
 def simulated_replay(args,check):
  assert check is True;path=pathlib.Path(args[args.index('--report')+1]);
  with path.open('x') as f:json.dump(r,f)
 records=[]
 for n in range(2):
  attempt,path,h=recheck(out,binding,bsha,probe,root/'probe-shards',longest,root/'coverage.py',sha,execute=simulated_replay);payload={'attempt_id':attempt,'probe_recheck_path':path,'probe_recheck_sha256':h};validate_attempt(payload,bsha,root,longest,False,sha);records.append(payload)
 assert records[0]['probe_recheck_path']!=records[1]['probe_recheck_path'];validate_attempt(records[0],bsha,root,longest,False,sha)
 binding_guard(binding,bsha,sha);binding.write_text(json.dumps({'binding':{'x':1},'longest_request':{'id':'changed','split':'test'}}))
 try:recheck(out,binding,bsha,probe,root/'probe-shards',longest,root/'coverage.py',sha,execute=simulated_replay)
 except ValueError:pass
 else:raise AssertionError('outer binding modification accepted')
try:require_fresh_probe(True,{'already_written_shard'})
except ValueError:pass
else:raise AssertionError('probe shard without receipt accepted for resource proof')
require_fresh_probe(False,{'completed_full_shard'})
print('ATTEMPT_RESUME_AND_OUTER_BINDING_CPU_TEST_PASS (simulated replay; no CUDA gate)')
