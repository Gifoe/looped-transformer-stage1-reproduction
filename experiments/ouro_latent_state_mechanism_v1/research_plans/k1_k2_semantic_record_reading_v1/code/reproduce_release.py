"""Verify public bytes and reproduce frozen scalar statistics with no torch/GPU/model."""
import pathlib,json,hashlib,gzip,tempfile,shutil,subprocess,sys,os
from assemble_release import assemble
P=pathlib.Path(__file__).resolve().parent.parent;assemble(P);manifest=json.loads((P/'PUBLICATION_MANIFEST.json').read_text())
for z in manifest['files']:
 d=(P/z['path']).read_bytes();assert len(d)==z['bytes'] and hashlib.sha256(d).hexdigest()==z['sha256'],z['path']
with tempfile.TemporaryDirectory(prefix='semantic_read_cpu_') as tmp:
 R=pathlib.Path(tmp);shutil.copytree(P/'code',R/'code')
 for pattern in ['*.json','*.md']:
  for f in P.glob(pattern):shutil.copyfile(f,R/f.name)
 for name in ['INPUTS.jsonl','ALL_QUESTION_RESULTS.jsonl','REFERENCE_PREFIX_PROVENANCE.jsonl']:
  with gzip.open(P/(name+'.gz'),'rb') as src,(R/name).open('wb') as out:shutil.copyfileobj(src,out)
 raw_hashes={z['path']:z['sha256'] for z in json.loads((P/'RAW_TENSOR_MANIFEST.json').read_text())['files']}
 rows=[json.loads(l) for l in (R/'ALL_QUESTION_RESULTS.jsonl').read_text().splitlines()];provenance=[json.loads(l) for l in (R/'REFERENCE_PREFIX_PROVENANCE.jsonl').read_text().splitlines()];(R/'native_results').mkdir();(R/'results').mkdir()
 for z in provenance:
  group=[u for u in rows if u['pair_id']==z['pair_id']];native=[{'id':u['id'],'native':u['conditions']['native'],'prefix_key':u['prefix_key']} for u in group];(R/'native_results'/(z['pair_id']+'.json')).write_text(json.dumps({**z,'rows':native}));science=[u for u in group if u['intervention_run']]
  if science:(R/'results'/(z['pair_id']+'.json')).write_text(json.dumps({'pair_id':z['pair_id'],'split':z['split'],'freeze_sha256':z['freeze_sha256'],'raw_sha256':raw_hashes[z['pair_id']+'_science.pt'],'rows':science}))
 env={**os.environ,'SEMANTIC_ROOT':str(R),'CUDA_VISIBLE_DEVICES':''};checked=[]
 for phase in ['development','confirmation']:
  for stage,name in [('native','NATIVE_CAPABILITY'),('science','SEMANTIC_SUMMARY')]:
   target=phase+'_'+name+'.json'
   if not(P/target).exists():continue
   # Relocate only the guard's workspace pointer, then execute original source bytes.
   bootstrap="import sys,pathlib,runpy;sys.path.insert(0,sys.argv[1]);import freeze_guard;freeze_guard.R=pathlib.Path(sys.argv[2]);sys.argv=[str(freeze_guard.R/'code/analyze.py'),sys.argv[3],sys.argv[4]];runpy.run_path(sys.argv[0],run_name='__main__')"
   subprocess.run([sys.executable,'-c',bootstrap,str(R/'code'),str(R),phase,stage],check=True,env=env);assert json.loads((R/target).read_text())==json.loads((P/target).read_text()),target;checked.append(target)
 receipt={'status':'PASS','public_bytes_verified':True,'original_frozen_source_input_model_hash_contract_verified':True,'exact_scalar_summaries':checked,'worlds_replayed':len(rows),'torch_or_GPU_used':False,'scope':'Scalar/statistical replay only; originalrawKVandfullheads are not publiclyredistributed; notindependentneuralreproduction','numpy':__import__('numpy').__version__,'python':sys.version};print(json.dumps(receipt,indent=2))
 if '--receipt' in sys.argv:pathlib.Path(sys.argv[sys.argv.index('--receipt')+1]).write_text(json.dumps(receipt,indent=2))
