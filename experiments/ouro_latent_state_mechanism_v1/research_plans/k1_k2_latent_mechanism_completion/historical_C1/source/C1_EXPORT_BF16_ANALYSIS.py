"""Export the completed BF16 evidence while numerical sensitivity is pending."""
import gzip,hashlib,json,tarfile
from pathlib import Path
R=Path('/root/autodl-tmp/ouro-k1-k2-mechanism')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(R/'C1_QUEUE_STATE.json')['status']=='COMPLETE'
for split in ['TRAIN','VALIDATION']:
 for suffix in ['COMPLETE.json','ALL_ENDPOINT_CPU_AUDIT.json','ALL_CONTROL_TENSOR_AUDIT.json']:
  assert read(R/'analysis'/(split+'_'+suffix))['status']=='PASS'
tables=read(R/'analysis/TRAIN_CONTRAST_TABLES.json')+read(R/'analysis/VALIDATION_CONTRAST_TABLES.json')
decisions=read(R/'analysis/TRAIN_DECISION_TABLES.json')+read(R/'analysis/VALIDATION_DECISION_TABLES.json')
result={'scope':'Completed BF16 interim; original FP32 attempt failed native-repeat identity and is being diagnosed. Not a final full mechanism claim.',
 'gate':read(R/'analysis/C1_ROUTE_GATE_VERDICT.json'),
 'primary':[x for x in tables if x['cohort'] in ['train1030','validation299'] and x['group']=='rescue' and x['sensitivity']=='all' and x['metric']=='meanwrong'],
 'rescue_decisions':[x for x in decisions if x['cohort'] in ['train1030','validation299'] and x['group']=='rescue' and x['sensitivity']=='all']}
(R/'BF16_INTERIM_FOR_ROOT.json').write_text(json.dumps(result,indent=2)+'\n')
files=[p for p in sorted((R/'analysis').rglob('*')) if p.is_file()]
manifest={'files':[{'path':str(p.relative_to(R)),'sha256':sha(p)} for p in files],
 'interim_sha256':sha(R/'BF16_INTERIM_FOR_ROOT.json')}
(R/'C1_BF16_ANALYSIS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=R/'C1_BF16_ANALYSIS_BUNDLE.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tf:
 for p in files:tf.add(p,arcname=str(p.relative_to(R)),recursive=False)
 for name in ['BF16_INTERIM_FOR_ROOT.json','C1_BF16_ANALYSIS_BUNDLE_MANIFEST.json']:
  tf.add(R/name,arcname=name,recursive=False)
print(json.dumps({'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files)}))
