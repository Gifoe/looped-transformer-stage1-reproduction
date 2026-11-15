"""Explicit allowlist package. Never copies model weights, credentials or raw CAS."""
import gzip,hashlib,json,pathlib,re,shutil
R=pathlib.Path(__file__).resolve().parents[1];C=pathlib.Path('/root/autodl-tmp/ouro-k1-k2-mechanism')
O=R/'publication/experiments/ouro_latent_state_mechanism_v1/research_plans/k1_k2_latent_mechanism_completion'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(pathlib.Path(p).read_text())
assert (O/'FINAL_RESEARCH_REPORT.md').exists()
assert read(O/'figures/VISUAL_QA.json')['status']=='PASS'
assert read(R/'B2_QUEUE_STATE.json')['status']=='COMPLETE' and read(R/'posthoc_review/QUEUE_STATE.json')['status']=='COMPLETE'
assert not (O/'PACKAGE_MANIFEST.json').exists()
origins=[]
def copy(source,relative,compressed=False):
 source=pathlib.Path(source);dest=O/relative
 assert source.is_file() and not dest.exists(),(source,dest)
 dest.parent.mkdir(parents=True,exist_ok=True)
 if compressed:
  with source.open('rb') as src,dest.open('xb') as raw:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=6,filename='') as output:shutil.copyfileobj(src,output)
 else:shutil.copy2(source,dest)
 origins.append(dict(file=str(dest.relative_to(O)),original_server_path=str(source),original_sha256=sha(source),gzip=compressed))
copy(R/'reporting/INTERACTION_SUPPLEMENT_FREEZE.json','receipts/reporting/INTERACTION_SUPPLEMENT_FREEZE.json')
# Exact executed original scientific tree and original freezes; no SSH helpers.
sources={}
for name in ['FULL_SOURCE_FREEZE.json','B2_FULL_SOURCE_FREEZE.json']:
 f=read(R/name)
 for rel,digest in f['sources'].items():
  assert sha(R/rel)==digest
  sources[rel]=digest
 sources[name]=sha(R/name)
for rel in sorted(sources):copy(R/rel,'source/original_tree/'+rel)
for folder in ['pipeline','posthoc_review','reporting','storage','bridge','async_io','parallel']:
 for p in sorted((R/folder).glob('*.py')):
  rel='source/execution_and_review/'+folder+'/'+p.name
  if not (O/rel).exists():copy(p,rel)
# Frozen orders and supplements not already present in the original tree.
for p in [R/'A_SIDE_SOURCE_FREEZE.json',R/'SELECTED_PAIR_SUPPLEMENT_FREEZE.json',R/'E2_ALL_STAGES_COMPLETE.json',R/'SELECTED_PAIR_ALL_COMPLETE.json',R/'AB1_HOLDOUT_SOL_INDEPENDENT_AUDIT.json',R/'NATIVE_GROUP_RECONSTRUCTION_AUDIT.json',R/'INSTALLER_CLEANUP_RECEIPT.json']:
 if p.exists():copy(p,'receipts/'+p.name)
for folder,names in {
 'async_io':['EXECUTION_AMENDMENT.json','REAL_STORAGE_EQUIVALENCE.json','SYNTHETIC_FAIL_CLOSED_QA.json','TRANSITION_INTENT.json','TRANSITION_RECEIPT.json','TRANSITION_SUPPLEMENT.json','FIRST_NEW3_TENSOR_AUDIT.json','THROUGHPUT_OBSERVATION.json'],
 'parallel':['EXECUTION_AMENDMENT.json','TRANSITION_INTENT.json','TRANSITION_RECEIPT.json','ACCEPTANCE_AND_LAUNCH.json','TECH_FREEZE.json','TECH_FREEZE_v2.json','TECH_FREEZE_v3.json','TECH_FREEZE_EMPTY_GPU.json','BENCHMARK_FAILURE.json','BENCHMARK_FAILURE_v2.json','BENCHMARK_FAILURE_v3.json','BENCHMARK_EMPTY_GPU_RESULT.json','V2_METADATA_DIFFERENCE_DIAGNOSIS.json','ALL_LINEAGE_COMPLETE.json']}.items():
 for name in names:
  p=R/folder/name
  if p.exists():
   compressed=p.stat().st_size>100000
   copy(p,'receipts/'+folder+'/'+name+('.gz' if compressed else ''),compressed=compressed)
for stage in ['train64','train1030','validation299']:
 for family,directory in [('AB1',R/'analysis_results'/stage),('B2',R/'b2_analysis'/stage)]:
  for p in sorted(directory.iterdir()):
   if not p.is_file() or p.suffix not in ['.json','.jsonl']:continue
   if p.name in ['rows.jsonl','summary.json','E2_ROWS.jsonl','E2_DESCRIPTIVE.json','A_VECTOR_GEOMETRY.jsonl','A_ANSWER_LABEL_GEOMETRY.jsonl','SELECTED_PAIR_VECTORS.json']:
    copy(p,'results/'+family+'/'+stage+'/'+p.name+'.gz',True)
   else:copy(p,'results/'+family+'/'+stage+'/'+p.name)
 for name in ['INDEPENDENT_REVIEW.json','BINARY_PATHS.json','compact_scalar_evidence.jsonl']:
  p=R/'posthoc_review'/stage/name;copy(p,'results/B2/'+stage+'/'+name+'.gz',True)
 copy(R/'parallel'/(stage+'_LINEAGE.json'),'receipts/parallel/'+stage+'_LINEAGE.json')
 # Case manifest and compact original metadata, excluding only enormous raw chunk maps.
 for family,directory in [('AB1',R/'data'/stage),('B2',R/'b2_data/treatments'/stage)]:
  for name in ['PENDING.json','MANIFEST_COVERAGE.json','INDEPENDENT_COVERAGE_AUDIT.json']:
   p=directory/name
   if p.exists():copy(p,'receipts/'+family+'/'+stage+'/'+name+'.gz',True)
  destination=O/'results'/family/stage/'metadata_without_raw_tensor_manifest.jsonl.gz'
  with destination.open('xb') as raw:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as stream:
    for case in read(directory/'PENDING.json')['cases']:
     p=directory/case['metadata_file'];assert sha(p)==case['metadata_sha256'];m=read(p)
     m.pop('tensor_manifest');m['original_metadata_file_sha256']=case['metadata_sha256']
     stream.write((json.dumps(m,allow_nan=False)+'\n').encode())
# Historical completed5090 C1, distinct from4090 AB1/B2.
for p in sorted(C.glob('C1_*.py')):copy(p,'historical_C1/source/'+p.name)
for name in ['branch_interfaces.py','branch_interfaces_v2.py','E2_FROZEN_DECODER.py','ASTRA_ORDER_C1_UPSTREAM_TO_L18.md','SOURCE_IMPLEMENTATION_AUDIT.md','BF16_BRANCH_ALIAS_CPU_AUDIT.json','AUDIT_BF16_BRANCH_ALIAS.py','AUDIT_PRIOR_L_ROWS.py','FP32_NATIVE_REPEAT_original_COMPLETE.json','FP32_NATIVE_REPEAT_deterministic_math_COMPLETE.json','E2_REUSE_ARTIFACT_AUDIT.md','E2_REUSE_ARTIFACT_INDEX.json','ASTRA_CAUSAL_GAP_REVIEW.md','RECURRENT_BOUNDARY_SOURCE_AUDIT.md','FP32_NATIVE_REPEAT_DIAGNOSIS.md','C1_FP32_MATH8_SOL_REVIEW.md','C1_FP32_MATH8_SOL_REVIEW.json','C1_EXECUTION_FREEZE.json']:
 if (C/name).exists():copy(C/name,'historical_C1/source/'+name)
for directory in ['analysis','evidence_packet']:
 for p in sorted((C/directory).iterdir()):
  if p.is_file() and (p.suffix in ['.json','.gz']):copy(p,'historical_C1/'+directory+'/'+p.name+('.gz' if p.suffix=='.json' else ''),p.suffix=='.json')
# Readable machine-generated short primary tables; full summaries retain all IDs.
for stage in ['train64','train1030','validation299']:
 s=read(R/'b2_analysis'/stage/'summary.json');primary={}
 for family,body in s['rescue']['families'].items():
  primary[family]=dict(all_positive_pass=body['all_positive_pass'],contrasts={name:{k:c[k] for k in ['n_defined','n_total','mean','ci95','lower','positive_pass','full_defined_coverage','natural_no_tie_mean']} for name,c in body['contrasts'].items()})
 p=O/'results/B2'/stage/'PRIMARY.json';p.write_text(json.dumps(dict(native_group_counts={k:v['n'] for k,v in s.items()},families=primary),indent=2))
# Two-MiB byte shards keep connected-GitHub blob writes bounded; originals remain reconstructable.
shards={}
for path in sorted(O.rglob('*')):
 if not path.is_file() or path.stat().st_size<=2*1024**2:continue
 relative=str(path.relative_to(O));original_sha=sha(path);length=path.stat().st_size;parts=[];digest=hashlib.sha256()
 with path.open('rb') as h:
  index=0
  while True:
   raw=h.read(2*1024**2)
   if not raw:break
   part=path.parent/(path.name+f'.part{index:04d}')
   with part.open('xb') as out:out.write(raw)
   assert part.read_bytes()==raw;digest.update(raw);parts.append(dict(path=str(part.relative_to(O)),bytes=len(raw),sha256=sha(part)));index+=1
 assert digest.hexdigest()==original_sha and sum(x['bytes'] for x in parts)==length
 shards[relative]=dict(bytes=length,sha256=original_sha,parts=parts)
 assert sha(path)==original_sha
 path.unlink()
(O/'SHARDED_ARTIFACTS.json').write_text(json.dumps(shards,indent=2))

# Final source/data integrity and credential guard. No private helpers are allowed.
for p in O.rglob('*'):
 if not p.is_file():continue
 assert 'remote_private' not in p.name and 'connection.json'!=p.name
 if p.suffix in ['.py','.md','.json','.jsonl','.txt','.sh']:
  content=p.read_text(errors='strict')
  assert not re.search(r'(?:hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_]{20,}',content),('Possible credential literal; do not publish',str(p.relative_to(O)))
files=[]
for p in sorted(O.rglob('*')):
 if p.is_file():
  data=p.read_bytes();files.append(dict(path=str(p.relative_to(O)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),git_blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
manifest=dict(status='REVIEWABLE_PACKAGE_NOT_YET_PUBLISHED',files=files,origins=origins,source_science_freeze_sha256=sha(R/'B2_FULL_SOURCE_FREEZE.json'),scope='Complete code/compact scalar results and audit receipts. Model weights and full raw tensor CAS excluded and retained server-side. Figures require separate PASS visual QA. No credentials/private connection helpers included.')
(O/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print('PACKAGE_READY',len(files),sum(x['bytes'] for x in files))
