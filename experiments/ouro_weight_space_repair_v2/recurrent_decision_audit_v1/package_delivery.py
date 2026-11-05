"""Lossless audit delivery; no model computation or historical mutation."""
from pathlib import Path
import hashlib,json,tarfile,io
O=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()
def main():
 excluded={'official_server_cpu_replay.json':'transport coordinate wrapper; sanitized official_environment_exact_replay.json included','live_resource_cache_readonly.json':'transport wrapper; sanitized live_resource_cache_preflight.json included','runtime_metadata_raw.json':'transport wrapper; runtime_metadata.json included','a4_SERVER_PREFLIGHT.json':'transport wrapper; a4_SERVER_PREFLIGHT_PUBLIC.json included'}
 for p in O.glob('*_RAW.json'):excluded[p.name]='private transport-coordinate wrapper; corresponding scientific receipt is retained'
 for p in O.glob('*PRIVATE*'):excluded[p.name]='private coordination'
 exclusions=dict(private_wrappers=[dict(path=n,sha256=H((O/n).read_bytes()),reason=reason) for n,reason in excluded.items() if (O/n).is_file()],recovered_parent_evidence='recovered/ is an exact duplicate of publicR7/R10 archive members; restore_inputs.py plus inherited_recovery_receipt.json bind and recover these. Not repeated in the new archive.',upstream_assets='No model weights, tokenizer copy, upstream private source, credentials or host identifiers included.')
 (O/'DELIVERY_EXCLUSIONS.json').write_text(json.dumps(exclusions,indent=2)+'\n',encoding='utf8')
 generated={'audit_records.tar.xz','RECORDS_CONTENTS.json','DELIVERY_INDEX.json','DELIVERY_MANIFEST.json','PUBLISH_STATE.json'}
 paths=[p for p in O.iterdir() if p.is_file() and p.name not in excluded and p.name not in generated and not p.name.startswith('ROOT_ARCHIVE_VERIFICATION')]
 paths.sort(key=lambda p:p.name);members=[]
 for p in paths:
  b=p.read_bytes()
  members.append(dict(path=p.name,bytes=len(b),sha256=H(b)))
 archive=O/'audit_records.tar.xz';assert not archive.exists()
 with tarfile.open(archive,'w:xz',preset=9) as t:
  for p in paths:
   b=p.read_bytes();info=tarfile.TarInfo(p.name);info.size=len(b);info.mtime=0;info.mode=0o644;t.addfile(info,io.BytesIO(b))
 packed=archive.read_bytes();contents=dict(archive=archive.name,archive_bytes=len(packed),archive_sha256=H(packed),members=members,member_count=len(members),scope='Complete actual new audit records and code; inherited recoverable public evidence excluded without changing it.')
 (O/'RECORDS_CONTENTS.json').write_text(json.dumps(contents,indent=2)+'\n',encoding='utf8');index=dict(status='historical_CPU_audit_complete_A4_parity_only_64_not_started',archive=archive.name,archive_bytes=len(packed),archive_sha256=H(packed),contents_sha256=H((O/'RECORDS_CONTENTS.json').read_bytes()),restore='Verify whole archive SHA, enumerate only listed regular safe paths, extract into a separate replay audit directory, then verify each member SHA. Recover parent R7/R10 inputs via restore_inputs.py. Do not rerun the already-completed model attempt.',frozen_definition_commit='809c22e168a77cf779bd271a12e37e2dbec62ff3',old_parent_commit='014438171c78de389af52b8ab7c822db4add43a1',new_GPU_worker_seconds=54.93683083169162,total_GPU_worker_seconds=20889.326696651056,remaining_GPU_seconds=710.6733033489436)
 (O/'DELIVERY_INDEX.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf8');manifest=dict(status=index['status'],archive=contents,exclusions=exclusions,required_outputs=['REPORT_ZH.md','existing_score_parity.json','k1_k4_paired_transitions.csv','loopwise_predictions.csv','loopwise_rescue_damage.csv','permutation_stability.csv','depth_readout_coverage.json','hypothesis_evidence.md','compute_ledger.json'],actual_model_coverage='Only2freshparityquestions/8rotations/24bodies/72heads; no64or576collection',reproducible_code=[p.name for p in paths if p.suffix=='.py'])
 (O/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
 with tarfile.open(archive,'r:xz') as t:
  assert len(t.getmembers())==len(members)
  for m,e in zip(t,members):assert m.isfile() and m.name==e['path'] and H(t.extractfile(m).read())==e['sha256']
 print(json.dumps(dict(archive_bytes=len(packed),archive_sha256=H(packed),members=len(members))))
if __name__=='__main__':main()
