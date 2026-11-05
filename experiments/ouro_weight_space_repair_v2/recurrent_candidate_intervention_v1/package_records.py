"""Small lossless record archive, separate standard raw-vector xz parts."""
from pathlib import Path
import json,hashlib,tarfile,io
O=Path(__file__).resolve().parent;ROOT=O.parent
H=lambda b:hashlib.sha256(b).hexdigest()
def main():
 frozen=json.loads((O/'FROZEN_DEFINITION_MANIFEST.json').read_text(encoding='utf-8-sig'))
 for e in frozen['files']:assert H((ROOT/e['path']).read_bytes())==e['sha256'],e['path']
 excluded={'VECTORS.npz','VECTORS.npz.xz','records.tar.xz','RECORDS_CONTENTS.json','DELIVERY_INDEX.json','DELIVERY_MANIFEST.json','PUBLISH_STATE.json'}
 paths=[p for p in O.iterdir() if p.is_file() and p.name not in excluded and '.npz.xz.part' not in p.name and 'PRIVATE' not in p.name and not p.name.startswith('ROOT_ARCHIVE_VERIFICATION')]
 paths.sort(key=lambda p:p.name);records=[dict(path=p.name,source_path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=H(p.read_bytes())) for p in paths]
 inherited=json.loads((O/'EXECUTION_LOCK.json').read_text(encoding='utf8'))['inherited_ledgers']
 for n,h in inherited.items():
  p=ROOT/n;assert H(p.read_bytes())==h;records.append(dict(path='inherited_ledgers/'+n,source_path=n,bytes=p.stat().st_size,sha256=h))
 archive=O/'records.tar.xz';assert not archive.exists()
 with tarfile.open(archive,'w:xz',preset=9) as t:
  for e in records:
   b=(ROOT/e['source_path']).read_bytes();info=tarfile.TarInfo(e['path']);info.size=len(b);info.mtime=0;info.mode=0o644;t.addfile(info,io.BytesIO(b))
 whole=archive.read_bytes();contents=dict(archive=archive.name,archive_bytes=len(whole),archive_sha256=H(whole),members=records,member_count=len(records),inherited_ledger_copies='20 original receipts copied unchanged for audit, not additional attempts or cost; do not overwrite parent ledgers.')
 (O/'RECORDS_CONTENTS.json').write_text(json.dumps(contents,indent=2)+'\n',encoding='utf8');parts=json.loads((O/'VECTORS_PARTS_INDEX.json').read_text(encoding='utf8'))
 index=dict(status='complete24_native_input_intervention_STOP_C',records_archive=archive.name,records_bytes=len(whole),records_sha256=H(whole),contents_sha256=H((O/'RECORDS_CONTENTS.json').read_bytes()),vector_parts_index='VECTORS_PARTS_INDEX.json',vector_parts_index_sha256=H((O/'VECTORS_PARTS_INDEX.json').read_bytes()),vector_container_sha256=parts['whole_sha256'],vector_raw_sha256=parts['restored_sha256'],vector_raw_bytes=parts['restored_bytes'],raw_vector_count=1744,definition_commit='6168ef44d74d0715059e884df826fff75e5a0a67',parent_public_blobs_retained=2272,recovery='restore_delivery.py verifies all ordered parts, stream-decodes exact NPZ, validates every safe regular records member and refuses mismatched existing files. Independent CPU replay: root_postrun_replay.py, no private transport/models required.')
 (O/'DELIVERY_INDEX.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf8');manifest=dict(status=index['status'],records=contents,vectors=parts,required_outputs=['REPORT_ZH.md','intervention_protocol.json','prompt_and_target_audit.json','permutation_transitions.csv','distractor_effects.csv','content_letter_readout.csv','question_level_statistics.csv','hypothesis_comparison.md','numerical_validation.json','compute_ledger.json'],privacy='No model weights/tokenizer/upstream private source/host identity/transport credentials. All source/current data are in records or standard vector parts; parent public bytes retained.',GPU_budget=json.loads((O/'compute_ledger.json').read_text(encoding='utf8')))
 (O/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
 with tarfile.open(archive,'r:xz') as t:
  assert len(t.getmembers())==len(records)
  for m,e in zip(t,records):assert m.isfile() and m.name==e['path'] and H(t.extractfile(m).read())==e['sha256']
 print(json.dumps(dict(records_bytes=len(whole),records_sha256=H(whole),members=len(records),vector_parts=len(parts['parts']),raw_vectors_sha256=parts['restored_sha256'])))
if __name__=='__main__':main()
