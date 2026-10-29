"""Read-only checkpoint integrity and effective dependency provenance audit."""
import runtime_bootstrap
import json,hashlib,importlib.metadata,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];model=ROOT.parent/'looped_stage1_trajectory_audit_seed2021_v1/models/Ouro-1.4B'
versions={}
for package,module in [('torch','torch'),('transformers','transformers'),('lm_eval','lm_eval'),('datasets','datasets'),('numpy','numpy'),('scipy','scipy'),('scikit-learn','sklearn'),('tokenizers','tokenizers'),('huggingface_hub','huggingface_hub'),('safetensors','safetensors')]:
 versions[package]=dict(version=importlib.metadata.version(package),origin=importlib.util.find_spec(module).origin)
metadata=json.loads((model/'snapshot_metadata.json').read_text());reference=json.loads((ROOT/'sources/checkpoint_official_blobs.json').read_text());checks=[]
assert reference['metadata']['sha']==metadata['sha']
for item in reference['metadata']['siblings']:
 p=model/item['rfilename']
 if not p.is_file():continue
 if 'lfs' in item:
  with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  expected=item['lfs']['sha256'];kind='lfs_sha256'
 else:
  contents=p.read_bytes();digest=hashlib.sha1(b'blob '+str(len(contents)).encode()+b'\0'+contents).hexdigest();expected=item['blobId'];kind='git_blob_sha1'
 checks.append(dict(file=p.name,digest=digest,expected=expected,digest_kind=kind,matches=digest==expected,bytes=p.stat().st_size))
assert any(x['file']=='model.safetensors' for x in checks)
assert checks and all(x['matches'] for x in checks),checks
assert metadata['sha']=='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
(ROOT/'results/ENVIRONMENT_AND_CHECKPOINT_AUDIT.json').write_text(json.dumps(dict(passed=True,revision=metadata['sha'],reference_transport=reference['url'],dependencies=versions,checkpoint_files=checks),indent=2),encoding='utf8')
print(json.dumps(dict(passed=True,checkpoint_files=len(checks),versions={k:v['version'] for k,v in versions.items()})),flush=True)
