"""Local CPU-only development workload count. No model, score, outcome or
heldout input. This is resource planning, not expansion of the frozen pilot.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


def main():
    from tokenizers import Tokenizer
    from r28_prefix_task_inventory import selected, pair, Adapter
    from r7_references import prefix_alignment

    started = time.monotonic()
    tp = ROOT/'artifacts/r24_TOKENIZER_AUDIT_PRIVATE.json'
    assert hashlib.sha256(tp.read_bytes()).hexdigest() == 'fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d'
    tok = Adapter(Tokenizer.from_file(str(tp)))
    references = json.loads((ROOT/'configs/r7_reference_manifest.json').read_text())['references']
    seen = {n: {} for n in ['Q', 'N0', 'N1', 'N2']}
    text_checks = {n: set() for n in ['N0', 'N1', 'N2']}
    failures = []
    qcount = candidates = uses = 0
    ids_seen = set()
    def keep(context, ids):
        key = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        if key in seen[context]:
            assert seen[context][key] == len(ids)
        else:
            seen[context][key] = len(ids)
    for split, expected in [('discovery', 384), ('selection', 192)]:
        count = 0
        with (ROOT/('data/slcd_'+split+'.jsonl')).open(encoding='utf-8') as f:
            for line in f:
                row = selected(line, {'id', 'request_pairs'})
                assert row['id'] not in ids_seen
                ids_seen.add(row['id'])
                count += 1
                qcount += 1
                for candidate, request in enumerate(row['request_pairs']):
                    candidates += 1
                    qp = pair(tok, *request)
                    rp = [pair(tok, references[n], request[1]) for n in ['N0', 'N1', 'N2']]
                    mapping, reason = prefix_alignment(tok, request, qp, rp)
                    if mapping is None:
                        failures.append(dict(id=row['id'], candidate=candidate, reason=reason))
                        continue
                    for j in range(len(qp[1])):
                        uses += 1
                        past = tok.decode(qp[1][:j])
                        qids = qp[0]+qp[1][:j]
                        if tok.encode(request[0].rstrip()+past) != qids:
                            failures.append(dict(id=row['id'], candidate=candidate, position=j, reason='Q_independent_consumed_text_boundary'))
                        keep('Q', qids)
                        for k, name in enumerate(['N0', 'N1', 'N2']):
                            refids = rp[k][0]+rp[k][1][:mapping[k][j]]
                            if past not in text_checks[name]:
                                if tok.encode(references[name].rstrip()+past) != refids:
                                    failures.append(dict(id=row['id'], candidate=candidate, position=j, context=name, reason='reference_independent_consumed_text_boundary'))
                                text_checks[name].add(past)
                            keep(name, refids)
                    # Complete N0 likelihoods are required even if mapping has extra positions.
                    for j in range(len(rp[0][1])):
                        keep('N0', rp[0][0]+rp[0][1][:j])
        assert count == expected
    by = {n: dict(unique_inputs=len(v), input_tokens=sum(v.values()),
                  min_length=min(v.values()), max_length=max(v.values())) for n, v in seen.items()}
    base_calls = by['Q']['unique_inputs'] + by['N0']['unique_inputs'] + 2*(by['N1']['unique_inputs']+by['N2']['unique_inputs'])
    weighted = 5*by['Q']['input_tokens'] + 4*by['N0']['input_tokens'] + 8*(by['N1']['input_tokens']+by['N2']['input_tokens'])
    out = dict(status='complete_CPU_resource_inventory_no_model', questions=qcount,
               candidates=candidates, aligned_Q_position_uses=uses, by_context=by,
               unique_base_forward_count=base_calls, depth_weighted_input_tokens=weighted,
               execution_boundary_counterexamples=failures,
               boundary_admission=(not failures), GPU_worker_seconds=0,
               CPU_wall_seconds=time.monotonic()-started,
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               dataset_sha256={s: hashlib.sha256((ROOT/('data/slcd_'+s+'.jsonl')).read_bytes()).hexdigest() for s in ['discovery', 'selection']},
               scope='Only development384+192 input text/token counts; no gold, predictions, score or model. No confirmation/validation/test rows opened. Counts do not authorize a forward, new method, fallback or changes to the fixed24 pilot; all input counts are a conservative no-prior-state-reuse workload.')
    with (ROOT/'results/r28_ROOT_DEV576_RESOURCE_INVENTORY.json').open('x', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
        f.write('\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
