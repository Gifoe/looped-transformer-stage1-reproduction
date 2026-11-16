"""Whitelist release; exact binary copies preserve original frozen source hashes."""
import pathlib,json,hashlib,shutil,gzip,ast,datetime
R=pathlib.Path('/root/ouro-semantic-read-sol-20261117');A=pathlib.Path('/root/ouro-next-mechanism-plan-20261117');P=R/'publication';P.mkdir(exist_ok=True);RAW=pathlib.Path('/root/autodl-tmp/ouro-semantic-read-sol-20261117/raw')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for phase in ['development','confirmation']:
 if (R/(phase+'_SEMANTIC_SUMMARY.json')).exists():assert json.loads((R/(phase+'_FULL_VECTOR_CPU_AUDIT.json')).read_text())['status']=='PASS'
assert json.loads((R/'V2R1_TECHNICAL_GATE.json').read_text())['status']=='PASS'
rawfiles=[{'path':f.name,'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(RAW.glob('*.pt'))];(R/'RAW_TENSOR_MANIFEST.json').write_text(json.dumps({'files':rawfiles,'total_bytes':sum(z['bytes'] for z in rawfiles),'retention':'Authorizedserveronly; allrawfull49152heads plusoldtechnicalactualQKV/SDPA/projectionwitnesses; notweights; publiccandidatefullscoresandfullvectorhashesretained','scope':'Technicalwitnesses4worlds/all8sites; sciencefullheadsalldata, perrowscalarauditsallinputs. No claim thatallscientificinternalQKVvectorswereindependentlyrecomputed.'},indent=2))
sources=[]
for f in sorted((R/'code').glob('*.py')):ast.parse(f.read_text());sources.append({'path':str(f.relative_to(R)),'bytes':f.stat().st_size,'sha256':sha(f)})
(R/'FINAL_SOURCE_MANIFEST.json').write_text(json.dumps({'files':sources,'original_science_freeze_preserved':True,'parallel_AST_amendment':'PARALLEL_EXECUTION_AMENDMENT.json andexecution_sources; only groupfilter/completionfilenames adapted','V1failure_preserved':True},indent=2))
files=['README.md','SEMANTIC_RECORD_RESULTS.md','FINAL_MECHANISM_STATUS.md','NUMERICAL_OPERATOR_V1_FAILURE_AND_V2.md','COMPUTE_LEDGER.json','FROZEN_INPUTS_AND_SELECTION.json','GROUPS.json','GENERATOR_AUDIT.json','INPUT_CPU_AUDIT.json','SOURCE_RUNTIME.json','SCIENCE_FREEZE.json','TECHNICAL_FREEZE.json','TECHNICAL_NUMERICAL_GATE.json','V2_TECHNICAL_FREEZE.json','V2_CPU_ORACLE_FREEZE.json','V2_TECHNICAL_GPU.json','V2_CPU_ORACLE_AUDIT.json','V2_TECHNICAL_GATE.json','V2R1_TECHNICAL_FREEZE.json','V2R1_TECHNICAL_GPU.json','V2R1_CPU_ORACLE_AUDIT.json','V2R1_TECHNICAL_GATE.json','ASTRA_PROTOCOL.md','ASTRA_V2_PROTOCOL.md','PARALLEL_EXECUTION_AMENDMENT.json','RAW_TENSOR_MANIFEST.json','FINAL_SOURCE_MANIFEST.json']
for p in ['development','confirmation']:
 for tail in ['NATIVE_CAPABILITY','SEMANTIC_SUMMARY','FULL_VECTOR_CPU_AUDIT','NATIVE_COMPLETE','SCIENCE_COMPLETE']:
  name=p+'_'+tail+'.json'
  if(R/name).exists():files.append(name)
if (R/'PROSPECTIVE_STOP.json').exists():files.append('PROSPECTIVE_STOP.json')
for name in files:shutil.copyfile(R/name,P/name)
for name in ['EVIDENCE_AND_DUPLICATION_AUDIT.md','EVIDENCE_AUDIT_RECEIPT.json','ASTRA_IDENTIFICATION_REVIEW.md','ASTRA_ISO_IDENTIFIABILITY_AND_NEXT_GATE.md','ASTRA_PAIRED_MESSAGE_V2_PROTOCOL.md','DOCUMENT_INDEX.json','REPOSITORY_TREE_AUDIT.json']:
 if(A/name).exists():shutil.copyfile(A/name,P/name)
for folder in ['code','execution_sources']:
 for f in (R/folder).glob('*'):
  if f.is_file() and f.suffix in ['.py','.json']:
   dest=P/f.relative_to(R);dest.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(f,dest)
(P/'logs').mkdir(exist_ok=True)
for f in R.glob('*.log'):
 d=f.read_bytes();assert b'Authorization:' not in d and b'connect.bjb1' not in d;shutil.copyfile(f,P/'logs'/f.name)
for f in (R/'parallel_logs').glob('*.log'):shutil.copyfile(f,P/'logs'/f.name)
for name in ['INPUTS.jsonl','ALL_QUESTION_RESULTS.jsonl','REFERENCE_PREFIX_PROVENANCE.jsonl']:
 with (R/name).open('rb') as src,(P/(name+'.gz')).open('wb') as target:
  with gzip.GzipFile(filename='',mode='wb',fileobj=target,compresslevel=6,mtime=0) as out:shutil.copyfileobj(src,out)
artifacts=[]
for f in sorted(P.glob('*.gz')):
 data=f.read_bytes()
 if len(data)<=2*1024*1024:continue
 parts=[]
 for i,off in enumerate(range(0,len(data),2*1024*1024)):
  p=f.with_name(f.name+f'.part{i:03d}');p.write_bytes(data[off:off+2*1024*1024]);parts.append(p.name)
 artifacts.append({'path':f.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'parts':parts});f.unlink()
(P/'SHARDED_ARTIFACTS.json').write_text(json.dumps({'artifacts':artifacts,'assembly':'automatic via code/assemble_release.py'},indent=2));manifest=[]
for f in sorted(P.rglob('*')):
 if f.is_file() and '__pycache__' not in f.parts and f.name!='PUBLICATION_MANIFEST.json':
  d=f.read_bytes();manifest.append({'path':str(f.relative_to(P)),'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest()})
(P/'PUBLICATION_MANIFEST.json').write_text(json.dumps({'files':manifest,'total_bytes':sum(z['bytes'] for z in manifest),'starting_commit':'fa9bb5bf1794cb11e7d0b504747c13c304fc5d00','goal_status':'active,incompletemechanism'},indent=2));print('PUBLICATION_PACKAGED',len(manifest)+1,sum(z['bytes'] for z in manifest))
