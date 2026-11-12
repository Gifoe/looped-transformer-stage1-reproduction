"""Fail-closed source freezing and PQR completion primitives, CPU only."""
import hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(paths,bindings,runtime):
 if len(paths)!=len(set(map(str,paths))) or not paths:raise ValueError('paths')
 return {'schema':'ABCA_FREEZE_V1','files':{str(Path(p).resolve()):sha(p) for p in paths},'natural_and_model_binding':bindings,'runtime':runtime}
def verify(manifest,paths,bindings,runtime):
 if manifest!=freeze(paths,bindings,runtime):raise ValueError('source/runtime drift')
def pqr_complete(run,binding_sha,cohort):
 run=Path(run).resolve();status=json.loads((run/'status.json').read_text())
 if status['status']!='PQR_COMPLETE_PENDING_ASTRA_REVIEW':raise ValueError('PQR incomplete')
 analysis_path=Path(status['analysis_path']).resolve();analysis_path.relative_to(run)
 if sha(analysis_path)!=status['analysis_sha256']:raise ValueError('analysis hash')
 analysis=json.loads(analysis_path.read_text())
 if analysis['status']!='PQR_FULL_COVERAGE_DESCRIPTIVE_PENDING_REVIEW' or analysis['binding_sha256']!=binding_sha:raise ValueError('analysis binding')
 evidence={}
 for split,ids in cohort.items():
  ids=set(ids);n=len(ids);path=run/('coverage_'+split+'.json');j=json.loads(path.read_text());q=j['questions']
  if j['status']!='PQR_COMPLETE_CUDA_REPLAY_ZERO' or j['split']!=split or j['expected_questions']!=n or j['complete_questions']!=n or j['replayed_condition_logits']!=4*n or j['binding_sha256']!=binding_sha or len(q)!=n or {x['id'] for x in q}!=ids or len({x['id'] for x in q})!=n:raise ValueError('PQR coverage')
  if any(set(x)!={'id','metadata_sha256','logits_sha256'} or any(not isinstance(x[k],str) or len(x[k])!=64 for k in ('metadata_sha256','logits_sha256')) for x in q):raise ValueError('PQR shard fields')
  receipt=run/split/'_receipt.json';r=json.loads(receipt.read_text())
  if sha(receipt)!=j['receipt_sha256'] or r.get('status')!='PQR_SPLIT_PENDING_COVERAGE' or r.get('split')!=split or r.get('questions')!=n or r.get('binding_sha256')!=binding_sha:raise ValueError('PQR receipt')
  evidence[split]=sha(path)
 if analysis['coverage_sha256']!=evidence or any(analysis['splits'][s]['all']['n']!=len(cohort[s]) for s in cohort):raise ValueError('analysis coverage')
 return {'status_sha256':sha(run/'status.json'),'analysis_sha256':sha(analysis_path),'coverage_sha256':evidence}

def verify_replay_ledger(out,ledger,device):
 out=Path(out)
 if set(ledger)!={'A1','B','C','A2'}:raise ValueError('replay ledger count')
 for label,r in ledger.items():
  if r.get('scores_zero_difference') is not True or r.get('replay_device')!=device or r.get('receipt_sha256')!=sha(out/(label+'.json')) or r.get('tensor_sha256')!=sha(out/(label+'.safetensors')):raise ValueError('replay evidence drift')
