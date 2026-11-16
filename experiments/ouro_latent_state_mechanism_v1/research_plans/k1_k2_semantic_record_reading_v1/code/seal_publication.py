import pathlib,json,hashlib,shutil,sys,datetime
R=pathlib.Path('/root/ouro-semantic-read-sol-20261117');P=R/'publication';A=pathlib.Path('/root/ouro-next-mechanism-plan-20261117');final='--final' in sys.argv
for source in [R/'CPU_AUDIT_THREAD_AMENDMENT.json',A/'ASTRA_NEXT_DECISION_ZH.md']:
 if source.exists():shutil.copyfile(source,P/source.name)
shutil.copyfile(R/'code/seal_publication.py',P/'code/seal_publication.py')
for f in R.glob('*.log'):
 d=f.read_bytes();assert b'Authorization:' not in d and b'connect.bjb1' not in d;shutil.copyfile(f,P/'logs'/f.name)
if final:
 receipt=json.loads((R/'CPU_REPLAY_RECEIPT.json').read_text());assert receipt['status']=='PASS';shutil.copyfile(R/'CPU_REPLAY_RECEIPT.json',P/'CPU_REPLAY_RECEIPT.json');summary=json.loads((R/'confirmation_SEMANTIC_SUMMARY.json').read_text());status={'study_status':'COMPLETE','overall_research_goal':'ACTIVE, incomplete unique mechanism','scientific_confirmation_gate':summary['graph_directed_pairmessage_gate'],'stronger_K2':summary['stronger_K2_participation'],'worlds':2560,'scientific_body_forwards':12800,'fullvector_CPU_audit_endpoints':sum(json.loads((R/(phase+'_FULL_VECTOR_CPU_AUDIT.json')).read_text())['complete_saved_full49152_endpoints'] for phase in ['development','confirmation']),'public_CPU_replay':'PASS exact4summaries/frozenhashcontracts','V1_rawSDPApermutation':'FAIL_PERMUTATION_NULL preserved','currentoperator':'V2R1 idealFP64 pairedmessage on actualnativeBF16base; notV1literalnativeKVrecompute','executors':'Astra scientific design/review; Sol implementation, GPUexecution andCPU self-audits. No independent neuralreplication.','sealed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(P/'RELEASE_STATUS.json').write_text(json.dumps(status,indent=2));(R/'RELEASE_STATUS.json').write_bytes((P/'RELEASE_STATUS.json').read_bytes())
sources=[]
for f in sorted((P/'code').glob('*.py')):
 d=f.read_bytes();sources.append({'path':str(f.relative_to(P)),'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()})
(P/'FINAL_SOURCE_MANIFEST.json').write_text(json.dumps({'files':sources,'original_science_freeze_preserved':True,'parallel_AST_amendment':'PARALLEL_EXECUTION_AMENDMENT.json andexecution_sources; onlygroupfilter/completion filenamesadapted','V1failure_preserved':True,'packaging_repair':'initialpublicCPUtemporaryfolderomittedfrozenASTRAmd; copiedexactmd, no method/result/gatechanged; originalfailurelogkept'},indent=2));virtual={z['path'] for z in json.loads((P/'SHARDED_ARTIFACTS.json').read_text())['artifacts']};files=[]
for f in sorted(P.rglob('*')):
 if not f.is_file() or '__pycache__' in f.parts or f.name=='PUBLICATION_MANIFEST.json' or str(f.relative_to(P)) in virtual:continue
 d=f.read_bytes();files.append({'path':str(f.relative_to(P)),'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest()})
(P/'PUBLICATION_MANIFEST.json').write_text(json.dumps({'files':files,'total_bytes':sum(z['bytes'] for z in files),'starting_commit':'fa9bb5bf1794cb11e7d0b504747c13c304fc5d00','goal_status':'active,incompletemechanism'},indent=2));print('SEALED_FINAL' if final else 'PRE_REPLAY_PACKAGE_REPAIRED',len(files)+1,sum(z['bytes'] for z in files))
