from freeze_guard import verify, exact_ids, sha
FREEZE_SHA = verify()
import record_operator_v2r1 as O
from record_operator_v2r1 import torch
import pathlib, json, time, datetime, sys
R = O.ROOT
phase = sys.argv[1]
assert json.loads((R / 'V2R1_TECHNICAL_GATE.json').read_text())['status'] == 'PASS'
freeze = json.loads((R / 'SCIENCE_FREEZE.json').read_text())
qs = select_shard([q for q in map(json.loads, (R / 'INPUTS.jsonl').read_text().splitlines()) if q['split'] == phase])
outdir = R / 'native_results'
outdir.mkdir(exist_ok=True)
start = time.monotonic()
if phase == 'confirmation':
    development = json.loads((R / 'development_SEMANTIC_SUMMARY.json').read_text())
    assert development['graph_directed_pairmessage_gate'] and development['freeze_sha256'] == FREEZE_SHA
m = O.initialize()
for (j, pid) in enumerate(sorted({q['pair_id'] for q in qs})):
    path = outdir / (pid + '.json')
    if path.exists():
        prior = json.loads(path.read_text())
        assert prior['freeze_sha256'] == FREEZE_SHA
        exact_ids(prior['rows'], [q['id'] for q in qs if q['pair_id'] == pid])
        assert sha(O.RAW / (pid + ('_native.pt' if 'native_results' in str(path) else '_science.pt'))) == prior['raw_sha256']
        continue
    group = [q for q in qs if q['pair_id'] == pid]
    canonical = {}
    rows = []
    raw = {}
    for q in group:
        (z, heads, pref, _) = O.run(m, q, capture_prefix=True)
        key = q['prefix_ids_sha256']
        if key in canonical:
            assert pref == canonical[key]['hashes']
        else:
            canonical[key] = {'world': q['id'], 'hashes': pref}
        rows.append({'id': q['id'], 'native': z, 'prefix_key': key})
        raw[q['id']] = {d: x.cpu() for (d, x) in heads.items()}
    assert len(canonical) == 8
    torch.save(raw, O.RAW / (pid + '_native.pt'))
    path.write_text(json.dumps({'pair_id': pid, 'split': phase, 'rows': rows, 'canonical_native_prefixes': canonical, 'freeze_sha256': FREEZE_SHA, 'raw_sha256': sha(O.RAW / (pid + '_native.pt'))}))
    print('NATIVE_GROUP', phase, j + 1, len({q['pair_id'] for q in qs}), round(time.monotonic() - start, 1), flush=True)
(R / (phase + '_NATIVE_SHARD2OF3_COMPLETE.json')).write_text(json.dumps({'freeze_sha256': FREEZE_SHA, 'status': 'COMPLETE', 'worlds': len(qs), 'groups': len({q['pair_id'] for q in qs}), 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}))