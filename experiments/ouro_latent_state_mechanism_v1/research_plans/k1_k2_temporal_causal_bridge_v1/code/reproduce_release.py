"""Verify public bytes and replay all frozen statistics without CUDA/checkpoint."""
import pathlib,json,hashlib,gzip,sys,os,subprocess,tempfile
from assemble_release import assemble
P=pathlib.Path(__file__).resolve().parent.parent
assemble(P)
manifest=json.loads((P/'PUBLICATION_MANIFEST.json').read_text())
for z in manifest['files']:
 f=P/z['path'];d=f.read_bytes();assert len(d)==z['bytes'] and hashlib.sha256(d).hexdigest()==z['sha256'],z['path']
def strip(z):
 if isinstance(z,dict):return {k:strip(v) for k,v in z.items() if k!='utc'}
 if isinstance(z,list):return [strip(x) for x in z]
 return z
with tempfile.TemporaryDirectory(prefix='temporal_bridge_cpu_') as tmp:
 R=pathlib.Path(tmp)
 for name in ['INPUTS.jsonl','UNITS.jsonl','REFERENCE_KV_PROVENANCE.jsonl','ALL_QUESTION_RESULTS.jsonl']:
  with gzip.open(P/(name+'.gz'),'rb') as src,(R/name).open('wb') as dest:
   import shutil;shutil.copyfileobj(src,dest)
 for name in ['A_BACKGROUND_SELECTION.json','B_SELECTION.json']:(R/name).write_bytes((P/name).read_bytes())
 with (R/'ALL_QUESTION_RESULTS.jsonl').open() as a,(R/'B_QUESTION_RESULTS.jsonl').open('w') as b,(R/'C_QUESTION_RESULTS.jsonl').open('w') as c:
  for line in a:
   z=json.loads(line);b.write(json.dumps(z['region_B'])+'\n');c.write(json.dumps(z['path_C'])+'\n')
 env={**os.environ,'BRIDGE_ROOT':str(R),'CUDA_VISIBLE_DEVICES':''}
 for stage in ['A','B','C']:
  for phase in ['development','confirmation']:
   subprocess.run([sys.executable,str(P/'code'/('analyze_'+stage+'.py')),phase],env=env,check=True)
   name=phase+'_'+stage+'_SUMMARY.json';assert strip(json.loads((R/name).read_text()))==strip(json.loads((P/name).read_text())),name
 for script,outputs in [('audit_symmetry.py',['TEMPORAL_SYMMETRY_AUDIT.json']),('audit_scalars.py',['SCALAR_INPUT_CPU_AUDIT.json','RANDOM_DOSE_AUDIT.json'])]:
  subprocess.run([sys.executable,str(P/'code'/script)],env=env,check=True)
  for name in outputs:assert strip(json.loads((R/name).read_text()))==strip(json.loads((P/name).read_text())),name
 receipt={'status':'PASS','all_public_bytes_verified':True,'six_phase_summaries_exact':True,'reciprocal_identities_and_scalar_inputs_exact':True,'dose_distribution_exact':True,'scope':'CPU scalar replay; does not re-execute neural model or recover unavailable raw KV tensors','python':sys.version,'numpy':__import__('numpy').__version__}
 print(json.dumps(receipt,indent=2))
 if '--receipt' in sys.argv:pathlib.Path(sys.argv[sys.argv.index('--receipt')+1]).write_text(json.dumps(receipt,indent=2))
