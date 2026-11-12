"""ROUND4 input/source adapter draft. No execution entrypoint or GPU approval."""
import json,hashlib,sys
from pathlib import Path
from safetensors.torch import load_file
if not __debug__:raise RuntimeError('optimized Python forbidden')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def read_question(root,split,ident,expected_coverages,ledger):
 root=Path(root);run=root/'orchestration_runs/pqr-round3-20261111';old=root/'orchestration_runs/round2-full-20261111';orig=root/'orchestration_runs/run-20261107T053919Z-6d01138e';resume=root/'orchestration_runs/resume-test-20261111'
 if split not in ('train','validation','test') or ledger['id']!=ident or ledger['split']!=split:raise ValueError('identity')
 cv=run/('coverage_'+split+'.json');oc=old/('coverage_'+split+'.json');nc=(resume if split=='test' else orig)/('coverage_'+split+'.json')
 for name,p in (('pqr',cv),('round2',oc),('natural',nc)):
  if sha(p)!=expected_coverages[name]:raise ValueError('coverage drift')
 c=json.loads(cv.read_text());o=json.loads(oc.read_text());n=json.loads(nc.read_text())
 if c.get('status')!='PQR_COMPLETE_CUDA_REPLAY_ZERO' or c.get('split')!=split or c.get('complete_questions')!=c.get('expected_questions') or c.get('replayed_condition_logits')!=4*c['complete_questions']:raise ValueError('PQR coverage incomplete')
 def question(report):
  q=[x for x in report['questions'] if x['id']==ident]
  if len(q)!=1:raise ValueError('coverage ID')
  return q[0]
 cq,oq,nq=map(question,(c,o,n));key=hashlib.sha256(ident.encode()).hexdigest();nm=orig/'shards'/split/(key+'.json');pm=run/split/(key+'.json');pt=pm.with_suffix('.safetensors');om=old/split/(key+'.json');ot=om.with_suffix('.safetensors')
 for p,h in ((nm,nq['metadata_sha256']),(pm,cq['metadata_sha256']),(pt,cq['logits_sha256']),(om,oq['metadata_sha256']),(ot,oq['logits_sha256'])):
  if sha(p)!=h:raise ValueError('shard drift')
 sys.path.insert(0,str(root/'research_plans/prefix_question_partition_draft_20261111'))
 from partition import partition
 regions=partition(nm.read_bytes(),nq['metadata_sha256'],ledger);natural=json.loads(nm.read_text());records=sorted(natural['records'],key=lambda r:r['candidate']);pqr=json.loads(pm.read_text())['payload'];prior=pqr['result'];old_result=json.loads(om.read_text())['payload']['result']
 if pqr['id']!=ident or pqr['split']!=split or pqr['binding_sha256']!=c['binding_sha256'] or pqr['old_metadata_sha256']!=sha(om) or prior['regions']!=regions:raise ValueError('source chain')
 new=load_file(str(pt));old_t=load_file(str(ot));mapping={'211':new['corner_211'],'212':new['corner_212'],'221':old_t['core_K2C_K1R'],'222':old_t['core_K2C_K2R']}
 for corner,name in (('221','K2C_K1R'),('222','K2C_K2R')):
  if prior['corners'][corner]!=old_result['conditions'][name]:raise ValueError('old corner drift')
 ids=records[0]['encoding']['input_ids'];targets=[r['encoding']['target_ids'][0] for r in records]
 return {'id':ident,'split':split,'ids':ids,'targets':targets,'natural_scores':[[r['scores'][k] for r in records] for k in (0,1)],'regions':regions,'prior_logits':mapping,'prior_scores':{k:prior['corners'][k]['scores'] for k in mapping},'gold':prior['gold'],'fixed_wrong':prior['fixed_wrong'],'source_hashes':{'natural_metadata':sha(nm),'pqr_metadata':sha(pm),'pqr_logits':sha(pt),'round2_metadata':sha(om),'round2_logits':sha(ot)}}
