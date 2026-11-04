"""Postrun description of recorded R25 neighbors only; no model/tokenizer/gold."""
import collections, hashlib, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    paths = ['configs/r25_MEMORY_MANIFEST.json', 'results/r25_pilot.jsonl']
    memory = json.loads((ROOT / paths[0]).read_text())
    rows = [json.loads(s) for s in (ROOT / paths[1]).read_text().splitlines()]
    entries = memory['entries']
    values = memory['values']
    counts = collections.Counter(values)
    occ = {k: collections.Counter() for k in ['Q', 'N0', 'N1', 'N2']}
    agreement = {k: dict(same_occurrence=0, same_value=0) for k in ['N0','N1','N2']}
    positions = 0
    for row in rows:
        for candidate in row['candidates']:
            n = {r['context']: r['neighbor_occurrences'] for r in candidate['retrieval']}
            assert all(len(v) == candidate['tokens'] for v in n.values())
            positions += candidate['tokens']
            for k, v in n.items():
                occ[k].update(v)
            for k in agreement:
                agreement[k]['same_occurrence'] += sum(a == b for a,b in zip(n['Q'],n[k]))
                agreement[k]['same_value'] += sum(values[a] == values[b] for a,b in zip(n['Q'],n[k]))
    stats = {}
    for k, counter in occ.items():
        vcount = collections.Counter()
        dcount = collections.Counter()
        for i,n in counter.items():
            vcount[values[i]] += n
            dcount[entries[i]['demo_id']] += n
        stats[k] = dict(distinct_occurrences=len(counter), distinct_values=len(vcount),
                        distinct_demos=len(dcount), top_value_counts=vcount.most_common(10),
                        selected_value_frequency_mean=sum(counts[v]*n for v,n in vcount.items())/positions,
                        value_collision_probability=sum(n*n for n in vcount.values())/positions**2,
                        value_histogram=dict(sorted(vcount.items())),
                        occurrence_histogram=dict(sorted(counter.items())))
    out = dict(status='complete_postrun_descriptive_only',n=len(rows),positions=positions,
               input_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
               self_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
               memory_occurrences=len(entries),memory_types=len(counts),
               uniform_occurrence_expected_value_frequency=sum(n*n for n in counts.values())/len(entries),
               contexts=stats,Q_vs_reference_agreement=agreement,
               scope='Recorded neighbors only; no gold, targets, model, new keys/queries, or GPU. Posthoc description, not independent inference or method selection.')
    path=ROOT/'results/r25_ASTRA_OCCUPANCY.json'
    with path.open('x',encoding='utf8') as f:json.dump(out,f,indent=2)
    print(json.dumps({k: v for k,v in out.items() if k!='contexts'}))
    print(json.dumps({k:{a:b for a,b in v.items() if 'histogram' not in a} for k,v in stats.items()}))

if __name__ == '__main__':main()
