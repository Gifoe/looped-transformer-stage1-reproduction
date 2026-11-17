"""Whitelist code/scalars/audits; retain all full neural vectors on authorized server."""
import pathlib,json,gzip,hashlib,shutil,subprocess,sys,datetime,ast
R=pathlib.Path('/root/ouro-region-use-sol-20261117');RAW=pathlib.Path('/root/autodl-tmp/ouro-region-use-sol-20261117/raw');P=R/'publication';P.mkdir(exist_ok=True)
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as src:
  for b in iter(lambda:src.read(8388608),b''):h.update(b)
 return h.hexdigest()
for phase in ['development','confirmation']:assert json.loads((R/(phase.upper()+'_FULL_VECTOR_CPU_AUDIT.json')).read_text())['status']=='PASS'
selection=json.loads((R/'REGION_SELECTION.json').read_text());confirmation=json.loads((R/'REGION_CONFIRMATION.json').read_text());assert sha(R/'REGION_SELECTION.json')==json.loads((R/'FRESH_CONFIRMATION_FREEZE.json').read_text())['selection_sha256'];sel=selection['selected_mask'];assert sel==confirmation['selected_mask'];old=R.parent/'ouro-binding-interface-sol-20261117'
raw=[{'path':f.name,'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(RAW.glob('*.pt'))];(R/'RAW_TENSOR_MANIFEST.json').write_text(json.dumps({'files':raw,'bytes':sum(x['bytes'] for x in raw),'retention':'Complete49152-head vectors and technical actualQKV/SDPA/projection witnesses retained authorizedserver; publicscalars containtheir hashes. Scientific internalQKV notall saved.'},indent=2))
ledger={'GPU':'single RTX4090,3 concurrent Python workers','technical_forwards':36,'preflight_failed_before_model_forwards':0,'new_development_forwards':6144,'old_development_fullmask_forwards_reused':1024,'old_development_native_forwards_reused':512,'fresh_confirmation_forwards':2048*(3 if sel==7 else 5),'fresh_worlds':2048,'fresh_groups':64,'worlds_per_group':32,'new_saved_fullhead_endpoints':sum(json.loads((R/(p.upper()+'_FULL_VECTOR_CPU_AUDIT.json')).read_text())['saved_full49152_endpoints'] for p in ['development','confirmation'])+72,'executors':'Astra protocol/scientific-source review; Sol implementation, GPU execution and CPU numeric self-audits. No independent neural replication.'};(R/'COMPUTE_LEDGER.json').write_text(json.dumps(ledger,indent=2))
for f in R.glob('*.json'):
 if f.name.startswith('GITHUB_'):continue
 shutil.copyfile(f,P/f.name)
for f in R.glob('*.md'):shutil.copyfile(f,P/f.name)
for f in (R/'code').glob('*.py'):
 ast.parse(f.read_text());d=P/'code'/f.name;d.parent.mkdir(exist_ok=True);shutil.copyfile(f,d)
(P/'logs').mkdir(exist_ok=True)
for f in list(R.glob('*.log'))+list((R/'logs').glob('*.log')):
 data=f.read_bytes();assert b'Authorization:' not in data and b'connect.bjb1' not in data;shutil.copyfile(f,P/'logs'/f.name)
parent=[]
for name in ['record_operator.py','matched_gaussian.py','ab1_loader_draft.py']:
 f=R.parent/'ouro-semantic-read-sol-20261117'/'code'/name;d=P/'parent_sources'/name;d.parent.mkdir(exist_ok=True);shutil.copyfile(f,d);parent.append({'path':str(d.relative_to(P)),'source':str(f),'sha256':sha(f)})
(P/'PARENT_SOURCE_PROVENANCE.json').write_text(json.dumps({'parent_semantic_commit':'b743862f258db3a51a98d554200ef7327f277ae7','parent_discovery_commit':'12621805b8ffe9d274f01ff8fa8d7fd5f413d0e8','dependencies':parent,'all_inherited_hashes':'REGION_FREEZE.json, prior releases retain the exact files'},indent=2))
def gz(name,lines):
 with (P/name).open('wb') as dst:
  with gzip.GzipFile(filename='',mode='wb',fileobj=dst,mtime=0,compresslevel=6) as z:
   for line in lines:z.write(line)
gz('INPUTS.jsonl.gz',[(R/'INPUTS.jsonl').read_bytes()])
def rr(phase):
 for f in sorted((R/('results' if phase=='development' else 'confirmation_results')).glob('SR*.json')):
  for u in json.loads(f.read_text())['rows']:yield (json.dumps(u,separators=(',',':'))+'\n').encode()
 if phase=='development':
  for f in sorted((old/'results').glob('SR*_L11.json')):
   for u in json.loads(f.read_text())['rows']:yield (json.dumps({**u,'mask':7},separators=(',',':'))+'\n').encode()
for phase in ['development','confirmation']:gz(phase.upper()+'_ROWS.jsonl.gz',rr(phase))
art=[]
for f in list(P.glob('*.gz')):
 b=f.read_bytes()
 if len(b)<=2097152:continue
 parts=[]
 for i,j in enumerate(range(0,len(b),2097152)):
  d=f.with_name(f.name+f'.part{i:03d}');d.write_bytes(b[j:j+2097152]);parts.append(d.name)
 art.append({'path':f.name,'sha256':sha(f),'bytes':len(b),'parts':parts});f.unlink()
(P/'SHARDED_ARTIFACTS.json').write_text(json.dumps({'artifacts':art},indent=2))
for phase in ['development','confirmation']:
 cp=subprocess.run([sys.executable,str(P/'code/replay_region.py'),phase],text=True,capture_output=True);assert cp.returncode==0,(phase,cp.stdout,cp.stderr);print(cp.stdout,flush=True);shutil.copyfile(P/('PUBLIC_REPLAY_'+phase.upper()+'.json'),R/('PUBLIC_REPLAY_'+phase.upper()+'.json'))
status={'study':'COMPLETE_AUDITED','scientific_confirmation_gate':confirmation['status'],'selected_mask':sel,'region_map':{'1':'Qquestion','2':'Ooptionsandformatting','4':'Rfinaltoken'},'overall_goal':'ACTIVE_UNRESOLVED_FULL_COMPUTATION','scientific_count':ledger,'public_replay':'PASS development/Fraction selection + all18freshstatistics/bootstrapCI/gates','raw_vectors_public':False,'code_and_all_scalar_outcomes_public':True,'gate_failures':[k for k,v in confirmation['tests'].items() if not v],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(P/'RELEASE_STATUS.json').write_text(json.dumps(status,indent=2));(R/'RELEASE_STATUS.json').write_bytes((P/'RELEASE_STATUS.json').read_bytes())
files=[]
for f in sorted(P.rglob('*')):
 if not f.is_file() or '__pycache__' in f.parts or f.name=='PUBLICATION_MANIFEST.json':continue
 b=f.read_bytes();files.append({'path':str(f.relative_to(P)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()})
(P/'PUBLICATION_MANIFEST.json').write_text(json.dumps({'files':files,'total_bytes':sum(z['bytes'] for z in files),'starting_commit':'12621805b8ffe9d274f01ff8fa8d7fd5f413d0e8','goal':'active'},indent=2));print('REGION_RELEASE_PACKAGED',len(files)+1,sum(z['bytes'] for z in files),flush=True)
