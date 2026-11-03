"""Archive complete readable evidence; large BF16 tensors stay in their own bundle."""
import gzip,hashlib,json,re,zipfile
from runtime import ROOT,save

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

TEXT_EXTENSIONS={'.py','.md','.json','.jsonl','.csv','.txt','.yaml','.yml'}
SECRET_PATTERNS=[
 re.compile(r'(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,})'),
 re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
 re.compile(r'''(?im)["']?(?:password|passwd|access_token|refresh_token|api_key|client_secret)["']?\s*[:=]\s*["']([^"'\r\n]{8,})["']'''),
 re.compile(r'https?://[^\s/:@]+:[^\s/@]+@'),
]

def public_payload(path):
 payload=path.read_bytes();derived=False
 if path.relative_to(ROOT).as_posix()=='logs/ENVIRONMENT.json':
  # Only derived public bytes change; never overwrite the original audited receipt.
  def redact_roots(value):
   if isinstance(value,dict):
    return {k:('[local distribution root omitted]' if k=='distribution_root' else redact_roots(v)) for k,v in value.items()}
   if isinstance(value,list):return [redact_roots(v) for v in value]
   return value
  payload=json.dumps(redact_roots(json.loads(payload)),indent=2,ensure_ascii=False).encode('utf8');derived=True
 if path.suffix.lower() in TEXT_EXTENSIONS:
  text=payload.decode('utf-8-sig')
  for pattern in SECRET_PATTERNS:
   # Never include a matching credential in the exception or logs.
   if pattern.search(text):raise RuntimeError('Credential-like content requires exclusion from public archive: '+path.relative_to(ROOT).as_posix())
 return payload,derived
def main():
 audit=json.loads((ROOT/'ARTIFACT_AUDIT.json').read_text());assert audit['status']=='passed'
 assert json.loads((ROOT/'logs/FROZEN_EVALUATION_PIPELINE.json').read_text())['status']=='complete'
 for relative,expected in audit['output_sha256'].items():assert sha(ROOT/relative)==expected,('Audited artifact changed',relative)
 for name,expected in audit['script_sha256'].items():assert sha(ROOT/'scripts'/name)==expected,('Audited source changed',name)
 paths=[]
 names=['REPORT_ZH.md','failure_mechanisms.csv','layer_intervention_results.csv','weight_edit_manifest.json','evaluation_summary.csv','paired_transitions.csv','paired_statistics.csv','failure_cases.jsonl','shared_confirmation_summary.csv','ARTIFACT_AUDIT.json','MECHANISM_IDENTIFIABILITY.md','NOTICES.md','REPRODUCE.md']
 paths.extend(ROOT/p for p in names)
 for rel in ['configs','sources','astra','figures','data']:
  paths.extend(p for p in (ROOT/rel).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pt','.pyc'])
 for study in ['original_call_parity','discovery_original','layer_screen_original','shared_confirmation_original','frozen_selection','frozen_validation','frozen_test']:
  paths.extend(p for p in (ROOT/'results'/study).rglob('*') if p.is_file() and p.suffix!='.pt')
 for p in ['logs/gpu_budget.json','logs/ENVIRONMENT.json','logs/FROZEN_EVALUATION_PIPELINE.json','logs/ORIGINAL_RESTART_PIPELINE.json','results/REPORT_ASSEMBLY.json','results/EVALUATION_ANALYSIS.json','artifacts/ACCEPTED_DIAGNOSTICS_MANIFEST.json']:
  paths.append(ROOT/p)
 paths.extend((ROOT/'scripts').glob('*.py'))
 paths=sorted(set(paths));out=ROOT/'artifacts/public_results.zip';assert not out.exists()
 out.parent.mkdir(exist_ok=True)
 # Preflight all members before creating an immutable final filename, so a guard
 # failure cannot leave a misleading complete-looking ZIP that blocks a retry.
 for p in paths:
  assert p.resolve().is_relative_to(ROOT.resolve()) and p.is_file()
  assert p.suffix.lower() not in {'.key','.pem','.p12','.pfx','.env'} and not p.name.startswith('.env')
  public_payload(p)
 temporary=out.with_suffix('.zip.partial');assert not temporary.exists()
 members=[]
 with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p in paths:
   assert p.resolve().is_relative_to(ROOT.resolve()) and p.is_file()
   rel=p.relative_to(ROOT).as_posix();payload,derived=public_payload(p);z.writestr(rel,payload)
   members.append(dict(path=rel,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),source_sha256=sha(p),derived_public_copy=derived))
  notes='This archive includes executable experiment scripts. Source and archived payload hashes are listed separately in PUBLIC_RESULTS_MANIFEST.json. Only logs/ENVIRONMENT.json is a derived public copy with distribution_root values omitted; its original receipt remains unchanged. Legitimate project paths are provenance, not credentials. Superseded explicit-mask evidence remains labeled engineering-only. No model weights or BF16 trace tensors are included.\n'.encode('utf8')
  z.writestr('PUBLIC_COPY_NOTES.txt',notes)
  members.append(dict(path='PUBLIC_COPY_NOTES.txt',bytes=len(notes),sha256=hashlib.sha256(notes).hexdigest(),source_sha256=None,derived_public_copy=False,generated=True))
 with zipfile.ZipFile(temporary) as z:assert z.testzip() is None
 temporary.replace(out)
 # Separate compressed CSV is convenient for readers who need only paired data.
 paired=ROOT/'artifacts/paired_transitions.csv.gz'
 with paired.open('xb') as raw:
  with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=9,mtime=0) as z:z.write((ROOT/'paired_transitions.csv').read_bytes())
 save('artifacts/PUBLIC_RESULTS_MANIFEST.json',dict(status='complete',archive=out.name,archive_bytes=out.stat().st_size,archive_sha256=sha(out),paired_csv_gzip_bytes=paired.stat().st_size,paired_csv_gzip_sha256=sha(paired),source_artifact_audit_sha256=sha(ROOT/'ARTIFACT_AUDIT.json'),credential_pattern_preflight=True,derived_public_members=[m['path'] for m in members if m['derived_public_copy']],generated_members=['PUBLIC_COPY_NOTES.txt'],large_bf16_vectors='Separate accepted_diagnostics.tar.gz; verify its own manifest independently; not in Git blobs',members=members))
 print('PUBLIC_RESULTS_PACKAGED',out.stat().st_size,'bytes',len(members),'members',flush=True)
if __name__=='__main__':main()
