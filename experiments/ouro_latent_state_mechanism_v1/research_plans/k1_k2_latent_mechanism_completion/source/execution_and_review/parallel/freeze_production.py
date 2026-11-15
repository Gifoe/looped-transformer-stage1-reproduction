import hashlib,json,pathlib,time
R=pathlib.Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(pathlib.Path(p).read_text())
gate=R/'parallel/BENCHMARK_EMPTY_GPU_RESULT.json';failure=R/'parallel/BENCHMARK_EMPTY_GPU_FAILURE.json'
if gate.exists():
 b=read(gate);assert b['status']=='PASS';workers=2 if b['production_candidate'] else 1
else:
 b=read(failure);assert any(p['workers']==1 for p in b['completed_phases']);workers=1
sf=R/'B2_FULL_SOURCE_FREEZE.json';s=read(sf)
assert all(sha(R/k)==v for k,v in s['sources'].items())
files=['parallel/worker_v3.py','parallel/parallel_storage.py','parallel/run_scientific_queue.py','parallel/verify_lineage_v2.py','parallel/freeze_production.py','parallel/transition_and_accept.py','parallel/benchmark_empty_gpu.py','parallel/TECH_FREEZE_EMPTY_GPU.json']
files.append('parallel/BENCHMARK_EMPTY_GPU_RESULT.json' if gate.exists() else 'parallel/BENCHMARK_EMPTY_GPU_FAILURE.json')
sources={k:sha(R/k) for k in files}
old=read(R/'async_io/EXECUTION_AMENDMENT.json');assert all(sha(R/k)==v for k,v in old['sources'].items());sources.update(old['sources'])
sources['async_io/EXECUTION_AMENDMENT.json']=sha(R/'async_io/EXECUTION_AMENDMENT.json')
a=dict(status='ISOLATED_PROCESS_SCIENCE_EXECUTION',workers=workers,utc_epoch=time.time(),science_freeze_sha256=sha(sf),sources=sources,benchmark_speedup=b.get('observed_speedup_two_workers'),scope='Original prospective Final25 cohorts, directions, operators, model, dtype, SDPA and all audits unchanged. Isolated model/hook process per fixed input-only shard. Per-worker private CAS with byte-verified rebased manifests. Only object IDs and allocation addresses excluded from cross-process metadata equality; all numerical evidence exact. Stage final root manifests retained; redundant internal manifests losslessly gzipped before removal. All pretransition committed evidence retained with original hashes.',completion='All fixed shard records must commit and match coverage. PASS result files denote fully joined CPU writer and completed source checks; lingering interpreter cleanup can be terminated only after verified PASS, with original independent full-tensor audit mandatory.')
with (R/'parallel/EXECUTION_AMENDMENT.json').open('x') as h:json.dump(a,h,indent=2)
print('PARALLEL_EXECUTION_FROZEN',workers,sha(R/'parallel/EXECUTION_AMENDMENT.json'))
