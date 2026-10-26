"""Verify preserved207-row bytes and the independently retried next baseline."""
import argparse
import hashlib
import json
from common import ROOT, save

parser = argparse.ArgumentParser()
parser.add_argument('--allow-prefix', action='store_true')
args = parser.parse_args()
snapshot = json.loads((ROOT/'logs/recovery_validation2p6_20261026/RECOVERY_SNAPSHOT.json').read_text())
entry = snapshot['preserved_files']['trajectories/math_Ouro-2.6B_validation.jsonl']
data = (ROOT/'trajectories/math_Ouro-2.6B_validation.jsonl').read_bytes()
assert hashlib.sha256(data[:entry['bytes']]).hexdigest() == entry['sha256']
boundary = data.rfind(b'\n')+1
rows = [json.loads(line) for line in data[:boundary].splitlines()]
assert 208 <= len(rows) <= 900 and len({r['record_key'] for r in rows}) == len(rows)
if not args.allow_prefix:
    assert len(rows) == 900 and boundary == len(data)
probe = json.loads((ROOT/'results/native26_recovery_next_case_probe.json').read_text())
actual = next(r for r in rows if r['record_key'] == probe['record_key'])
for key in ['token_ids', 'text', 'prediction', 'correct', 'numeric_correct', 'condition',
            'cache_layers', 'actual_strength_min', 'actual_strength_max']:
    assert actual[key] == probe[key], key
assert [[s['prediction'] for s in step['by_depth']] for step in actual['token_trajectory']] == [
       [s['prediction'] for s in step['by_depth']] for step in probe['token_trajectory']]
exit_record = json.loads((ROOT/'logs/native26_probe_exit.json').read_text())
assert exit_record['exit_code'] == 0 and not exit_record['timeout']
result = dict(passed=True, audited_records=len(rows), complete_validation=(len(rows) == 900 and boundary == len(data)),
              preserved_prefix_records=207, preserved_prefix_bytes=entry['bytes'],
              preserved_prefix_sha256=entry['sha256'], next_probe_matches_formal=True,
              next_case_key=probe['record_key'], next_baseline_depth_argmax_exact=True,
              scope='Old207 records byte-identical; next formal baseline equals the independent probe, including per-depth argmax. Prefix mode does not certify900-row completion.')
name = 'native26_recovery_early_lineage' if args.allow_prefix else 'native26_recovery_lineage_check'
save(f'results/{name}.json', result)
print(json.dumps(result), flush=True)
