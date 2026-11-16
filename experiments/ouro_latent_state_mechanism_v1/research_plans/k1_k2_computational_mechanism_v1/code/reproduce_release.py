"""Recompute all published scalar summaries on CPU. Requires NumPy1.26.4, no torch/GPU."""
import argparse,pathlib,json,gzip,tempfile,shutil,os,subprocess,sys,collections
from assemble_release import assemble
p=argparse.ArgumentParser();p.add_argument('--release',default='.');a=p.parse_args();pub=pathlib.Path(a.release).resolve();assemble(pub)
def unpack(name):
 with gzip.open(pub/'data'/name,'rt') as f:return [json.loads(l) for l in f]
def original_fields(x):
 if isinstance(x,dict):
  if 'captured_K2_stage_R_sha256' in x:x['stage_R_sha256']=x.pop('captured_K2_stage_R_sha256')
  for v in x.values():original_fields(v)
 elif isinstance(x,list):
  for v in x:original_fields(v)
with tempfile.TemporaryDirectory(prefix='ouro-computation-replay-') as tmp:
 R=pathlib.Path(tmp)
 for d in ['analysis','data/A','data/B','data/C','prior/analysis_inputs','code']: (R/d).mkdir(parents=True,exist_ok=True)
 for name in ['A_FREEZE.json','A_EXPANSION_FREEZE.json','B_FREEZE.json','C_FREEZE.json','ARC_STRUCTURE_FREEZE.json']:shutil.copy2(pub/'freeze'/name,R/name)
 for name in ['analyze.py','analyze_C.py']:shutil.copy2(pub/'code'/name,R/'code'/name)
 for name,rel in [('B_INPUTS.jsonl.gz','B_INPUTS.jsonl'),('ANALYSIS_LABELS.jsonl.gz','prior/analysis_inputs/analysis_labels.jsonl')]:
  with gzip.open(pub/'inputs'/name,'rb') as f:(R/rel).write_bytes(f.read())
 for x in unpack('A_QUESTION_RESULTS.jsonl.gz'):
  original_fields(x);(R/'data/A'/f'{x["split"]}__{x["id"]}.json').write_text(json.dumps(x))
 groups=collections.defaultdict(list)
 for u in unpack('B_QUESTION_RESULTS.jsonl.gz'):original_fields(u);groups[u['pair_id']].append(u)
 for pid,units in groups.items():(R/'data/B'/f'{pid}.json').write_text(json.dumps({'base_pair_id':pid,'units':units}))
 for x in unpack('C_QUESTION_RESULTS.jsonl.gz'):original_fields(x);(R/'data/C'/f'{x["unit_id"]}.json').write_text(json.dumps(x))
 env=os.environ.copy();env['OURO_COMPUTE_ROOT']=str(R)
 subprocess.run([sys.executable,str(R/'code/analyze.py'),'both'],env=env,check=True);subprocess.run([sys.executable,str(R/'code/analyze_C.py')],env=env,check=True)
 for name in ['A_SUMMARY.json','B_SUMMARY.json','C_SUMMARY.json']:
  actual=json.loads((R/'analysis'/name).read_text());expected=json.loads((pub/'analysis'/name).read_text());assert actual==expected,name+' differs';print(name,'EXACT_RECOMPUTATION_PASS')
print('PUBLIC_CPU_REPRODUCTION_PASS')
