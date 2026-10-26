"""Verify preserved pre-crash bytes and exact next-case replay after completion."""
import hashlib
import json
from common import ROOT,read,save

snapshot=json.loads((ROOT/'logs/recovery_validation1p4_20261026/RECOVERY_SNAPSHOT.json').read_text())
entry=snapshot['preserved_files']['trajectories/math_Ouro-1.4B_validation.jsonl']
path=ROOT/'trajectories/math_Ouro-1.4B_validation.jsonl';data=path.read_bytes()
assert hashlib.sha256(data[:entry['bytes']]).hexdigest()==entry['sha256']
rows=read(path);assert len(rows)==900 and len({r['record_key'] for r in rows})==900
probe=json.loads((ROOT/'results/recovery_next_case_probe.json').read_text())
actual=next(r for r in rows if r['record_key']==probe['record_key'])
for key in ['token_ids','text','prediction','correct','numeric_correct','condition','cache_layers','actual_strength_min','actual_strength_max']:
    assert actual[key]==probe[key],key
exit_record=json.loads((ROOT/'logs/recovery_probe_exit_attempt2.json').read_text())
assert exit_record['exit_code']==0 and not exit_record['timeout']
save('results/recovery_lineage_check.json',dict(passed=True,complete_validation_records=900,
     preserved_prefix_records=snapshot['valid_complete_records'],preserved_prefix_bytes=entry['bytes'],
     preserved_prefix_sha256=entry['sha256'],next_probe_matches_formal=True,next_case_key=probe['record_key'],
     scope='Pre-crash835 records are byte-identical; independently retried next configuration equals the resumed formal record; complete1.4 validation count verified'))
print(json.dumps(dict(passed=True,records=900,preserved_prefix=835,next_probe_matches_formal=True)),flush=True)
