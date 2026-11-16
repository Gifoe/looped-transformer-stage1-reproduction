import pathlib,json,hashlib,gzip,shutil,ast,subprocess,sys,datetime
R=pathlib.Path('/root/ouro-binding-interface-sol-20261117');P=R/'publication';P.mkdir(exist_ok=True);OLD=pathlib.Path('/root/ouro-semantic-read-sol-20261117');RAW=pathlib.Path('/root/autodl-tmp/ouro-binding-interface-sol-20261117/raw');assert json.loads((R/'DISCOVERY_FULL_VECTOR_CPU_AUDIT.json').read_text())['status']=='PASS'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
raw=[{'path':f.name,'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(RAW.glob('*.pt'))];(R/'RAW_TENSOR_MANIFEST.json').write_text(json.dumps({'files':raw,'total_bytes':sum(z['bytes'] for z in raw),'retention':'fullheadandtechnicalwitnessfiles retained authorizedserver, not publicbinaryweights','not_all_scientific_QKV_saved':True},indent=2));files=['README.md','ACTIVE_RUN.md','WHOLE_Q_CUBE_UNEXECUTED.md','DISCOVERY_FREEZE.json','LAYER_SELECTION.json','TECHNICAL_GATE.json','TECHNICAL_GPU.json','DISCOVERY_FULL_VECTOR_CPU_AUDIT.json','RAW_TENSOR_MANIFEST.json','ASTRA_DECISION.md','ASTRA_DUAL_ISO_CUBE_DESIGN.md','ISO_PROTOCOL_FROZEN_CANDIDATE.md','STAGE_H_REGION_CAUSAL_PROTOCOL.md']+[f.name for f in R.glob('SHARD_*COMPLETE.json')]+[f.name for f in R.glob('SOURCE_RUNTIME*.json')]
for n in files:shutil.copyfile(R/n,P/n)
for f in (R/'code').glob('*.py'):
 ast.parse(f.read_text());d=P/'code'/f.name;d.parent.mkdir(exist_ok=True);shutil.copyfile(f,d)
parent=[]
for n in ['record_operator_v2r1.py','record_operator.py','matched_gaussian.py','ab1_loader_draft.py','freeze_guard.py']:
 f=OLD/'code'/n;d=P/'parent_sources'/n;d.parent.mkdir(exist_ok=True);shutil.copyfile(f,d);parent.append({'path':'parent_sources/'+n,'actual_source':str(f),'sha256':sha(f)})
(P/'PARENT_SOURCE_PROVENANCE.json').write_text(json.dumps({'parent_commit':'b743862f258db3a51a98d554200ef7327f277ae7','files':parent},indent=2))
for name in ['coordinator.log','audit_discovery.log']:
 d=P/'logs'/name;d.parent.mkdir(exist_ok=True);shutil.copyfile(R/name,d)
for f in (R/'logs').glob('*.log'):shutil.copyfile(f,P/'logs'/f.name)
with gzip.GzipFile(filename='',mode='wb',fileobj=(P/'DEV_INPUTS.jsonl.gz').open('wb'),mtime=0) as out:out.write((R/'DEV_INPUTS.jsonl').read_bytes())
with gzip.GzipFile(filename='',mode='wb',fileobj=(P/'DISCOVERY_ROWS.jsonl.gz').open('wb'),mtime=0) as out:
 for f in sorted((R/'results').glob('SR*_L*.json')):
  for u in json.loads(f.read_text())['rows']:out.write((json.dumps(u,separators=(',',':'))+'\n').encode())
arts=[]
for f in P.glob('*.gz'):
 data=f.read_bytes()
 if len(data)<=2*1024*1024:continue
 parts=[]
 for i,off in enumerate(range(0,len(data),2*1024*1024)):
  d=f.with_name(f.name+f'.part{i:03d}');d.write_bytes(data[off:off+2*1024*1024]);parts.append(d.name)
 arts.append({'path':f.name,'bytes':len(data),'sha256':sha(f),'parts':parts});f.unlink()
(P/'SHARDED_ARTIFACTS.json').write_text(json.dumps({'artifacts':arts},indent=2));cp=subprocess.run([sys.executable,str(P/'code/replay_discovery.py')],text=True,capture_output=True);assert cp.returncode==0,(cp.stdout,cp.stderr);receipt={'status':'PASS','stdout':cp.stdout,'executor':'Sol separate stdlib Fraction replay, not independent model replication','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'PUBLIC_CPU_REPLAY.json').write_text(json.dumps(receipt,indent=2));shutil.copyfile(R/'PUBLIC_CPU_REPLAY.json',P/'PUBLIC_CPU_REPLAY.json');(P/'RELEASE_STATUS.json').write_text(json.dumps({'discovery':'COMPLETE_AUDITED','selected_layer':11,'independent_new_confirmation':False,'overall_goal':'ACTIVE','scientific_forwards':8192,'technical_forwards':12,'public_replay':'PASS','whole_Q_cube':'UNEXECUTED'},indent=2));manifest=[]
for f in sorted(P.rglob('*')):
 if not f.is_file() or '__pycache__' in f.parts or f.name=='PUBLICATION_MANIFEST.json':continue
 d=f.read_bytes();manifest.append({'path':str(f.relative_to(P)),'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest()})
(P/'PUBLICATION_MANIFEST.json').write_text(json.dumps({'files':manifest,'total_bytes':sum(z['bytes'] for z in manifest),'starting_commit':'b743862f258db3a51a98d554200ef7327f277ae7','goal_status':'active'},indent=2));print('DISCOVERY_PACKAGED',len(manifest)+1,sum(z['bytes'] for z in manifest),cp.stdout,flush=True)
