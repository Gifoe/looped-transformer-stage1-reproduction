"""Independent reconstruction from published LPs/graph; no solver or model."""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import math
import struct
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def rat(x):
    return F(int(x['numerator']), int(x['denominator']))


def main():
    start = time.monotonic()
    lock = 'configs/r31_SHARED_PREFIX_ORACLE_LOCK.json'
    spec = read(lock)
    for group in ('source_sha256', 'input_sha256', 'inherited_GPU_ledgers'):
        for path, value in spec[group].items():
            assert sha(path) == value, path
    total = sum(read(path)['used_seconds'] for path in spec['inherited_GPU_ledgers'])
    assert len(spec['inherited_GPU_ledgers']) == 18 and total == 20755.46525352029
    sup = read('logs/r31_ORACLE_SUPERVISOR.json')
    assert sup['status'] == 'complete' and sup['exit_code'] == sup['worker_exit_code'] == 0
    assert not sup.get('timed_out', False) and not sup['overrun'] and 0 < sup['CPU_job_wall_seconds'] <= 180
    assert sup['GPU_worker_seconds'] == 0
    raw = read('results/r31_ORACLE_RAW.json')
    analysis = read('results/r31_ORACLE_ANALYSIS.json')
    complete = read('results/r31_ORACLE_COMPLETE.json')
    assert raw['status'] == complete['status'] == 'complete' and raw['fatal'] is complete['fatal'] is None
    assert sup['analysis_sha256'] == sha('results/r31_ORACLE_ANALYSIS.json')
    assert sup['complete_sha256'] == sha('results/r31_ORACLE_COMPLETE.json')
    assert complete['raw_sha256'] == analysis['raw_sha256'] == sha('results/r31_ORACLE_RAW.json')
    assert complete['protocol_sha256'] == raw['protocol_sha256'] == analysis['protocol_sha256'] == sha(lock)
    assert raw['platform'] == spec['solver_identity']['platform']
    assert raw['python'].split()[0] == spec['solver_identity']['python']
    graph = read('configs/r28_TRUE_PREFIX_TASK_INVENTORY.json')
    requests = read('configs/r28_SANITIZED_REQUESTS.json')['rows']
    scores = {row['id']: row for row in read('results/r30_SCORES.json')['rows']}
    known = {row['id']: row for row in read('results/r30_ANALYSIS.json')['predictions']}
    links, owners = {}, {}
    for use in graph['usages']:
        identity = use['id'], use['candidate'], use['Q_position']
        assert identity not in links
        key = use['tasks']['Q']
        links[identity] = key
        owners.setdefault(key, set()).add(use['id'])
    assert len(links) == 548 and len(owners) == 421 and all(len(v) == 1 for v in owners.values())
    assert len(raw['records']) == len(requests) == len(analysis['rows']) == len(scores) == 24
    tau, eps, subnormal = F(1, 100000000), F(1, 2**24), F(1, 2**149)
    outcomes, endpoint_count = [], 0
    for request, record, released in zip(requests, raw['records'], analysis['rows']):
        qid = request['id']
        assert qid == record['id'] == released['id']
        y = known[qid]['gold']
        assert y == record['gold']
        chains, pieces, constants, bounds = [], [], [], []
        for j, (candidate, saved) in enumerate(zip(request['candidates'], scores[qid]['candidates'])):
            off = saved['pieces']['Fixed']['target_log_probabilities']
            on = saved['pieces']['primary']['target_log_probabilities']
            n, chars, late = len(off), saved['characters'], F(saved['late'])
            assert n == len(on) == len(candidate['Q_target_ids']) and chars == candidate['characters'] > 0
            for value in off+on:
                assert math.isfinite(value) and value <= 0 and struct.unpack('<f', struct.pack('<f', value))[0] == value
            f, p = list(map(F, off)), list(map(F, on))
            chain = [links[qid, j, t] for t in range(n)]
            chains.append(chain); pieces.append((f, p))
            constants.append(sum(f, F(0))/chars+late)
            size = sum((max(abs(a), abs(b)) for a, b in zip(f, p)), F(0))
            assert n*eps < F(1, 2)
            gamma = (n-1)*eps/(1-(n-1)*eps)
            assert size+gamma*size+n*subnormal <= (2**24-1)*2**104
            bounds.append((gamma*size+n*subnormal)/chars+16*F(1, 2**52)*(size/chars+abs(late)+1))
        keys = sorted({k for chain in chains for k in chain})
        coeff = []
        for candidate, chain, (f, p) in zip(request['candidates'], chains, pieces):
            by_key = dict.fromkeys(keys, F(0))
            for key, a, b in zip(chain, f, p):
                by_key[key] += (b-a)/candidate['characters']
            coeff.append(by_key)
        active = [k for k in keys if any(row[k] != 0 for row in coeff)]
        assert active == record['keys'] and [k for k in keys if k not in active] == record['zero_keys']
        assert chains == record['chains']
        d = [[row[k] for k in active] for row in coeff]
        assert constants == list(map(rat, record['a'])) and bounds == list(map(rat, record['B']))
        assert d == [[rat(v) for v in row] for row in record['d']]
        for j, saved in enumerate(scores[qid]['candidates']):
            assert abs(constants[j]-F(saved['scores']['A1'])) <= bounds[j]
            assert abs(constants[j]+sum(d[j], F(0))-F(saved['scores']['primary'])) <= bounds[j]
            endpoint_count += 2
        proposal = record['proposal']
        for field, actual in [('numpy','numpy'), ('scipy','scipy'), ('HiGHS','HiGHS_version')]:
            assert proposal[actual] == spec['solver_identity'][field]
        assert proposal['thread_limit'] == 1 and proposal['MILP_time_limit_seconds'] == 3
        wrong = [j for j in range(4) if j != y]
        assert wrong == record['wrong_candidates']
        lower = upper = None
        bits = proposal['witness']
        if bits is not None:
            assert len(bits) == len(active) and all(type(v) is int and v in (0, 1) for v in bits)
            route = [a+sum((v*x for v, x in zip(row, bits)), F(0)) for a, row in zip(constants, d)]
            margins = [route[y]-route[j]-bounds[y]-bounds[j] for j in wrong]
            lower = min(margins)
            assert lower == rat(proposal['witness_margin_lower'])
            assert margins == list(map(rat, proposal['witness_pair_lowers']))
        if proposal['dual_weights'] is not None:
            weights = list(map(rat, proposal['dual_weights']))
            assert len(weights) == 3 and all(w >= 0 for w in weights) and sum(weights, F(0)) == 1
            base = sum((w*(constants[y]-constants[j]+bounds[y]+bounds[j]) for w, j in zip(weights, wrong)), F(0))
            column = [sum((w*(d[y][k]-d[j][k]) for w, j in zip(weights, wrong)), F(0)) for k in range(len(active))]
            upper = base+sum((max(v, F(0)) for v in column), F(0))
            assert upper == rat(proposal['upper'])
        reachable = lower is not None and lower > tau
        unreachable = upper is not None and upper <= tau
        assert not (reachable and unreachable)
        state = 'certified_reachable' if reachable else 'certified_unreachable_strict_margin' if unreachable else 'undetermined'
        assert state == proposal['classification'] == released['classification']
        preds = known[qid]['predictions']
        outcomes.append(dict(id=qid, classification=state,
                             R6_error=preds['R6_matched'] != y,
                             R6_historical_error=preds['R6_historical'] != y,
                             R6_only_gain=preds['R6_matched'] == y and preds['Fixed'] != y,
                             lower=None if lower is None else str(lower),
                             upper=None if upper is None else str(upper),
                             upper_strictly_negative=upper is not None and upper < 0))
    counts = {c:sum(o['classification'] == c for o in outcomes) for c in analysis['counts']}
    assert counts == analysis['counts'] and endpoint_count == 192
    errors = [o for o in outcomes if o['R6_error']]
    protected = [o for o in outcomes if o['R6_only_gain']]
    result = dict(status='passed_Root_independent_original_LP_graph_rational_reconstruction',
                  n=24, candidates=96, positions=548, prefixes=421, endpoint_enclosures=192,
                  counts=counts, R6_errors=len(errors),
                  R6_errors_certified_unreachable=sum(o['classification'] == 'certified_unreachable_strict_margin' for o in errors),
                  R6_error_upper_strictly_negative=sum(o['upper_strictly_negative'] for o in errors),
                  R6_only_gain_cases=protected, independent_rows=outcomes,
                  actual_solver_versions_match_lock=True, all18_GPU_ledgers_unchanged=True,
                  used_GPU_seconds=total, remaining_GPU_seconds=21600-total,
                  CPU_verification_wall_seconds=time.monotonic()-start, new_GPU_seconds=0, model_loaded=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Retrospective exact certificate verification on the same released 24 rows. No solver, model, inference selector, new labels or independent efficacy test.')
    with (ROOT/'results/r31_ROOT_CERTIFICATE_VERIFICATION.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'independent_rows'}))


if __name__ == '__main__':
    main()
