"""Independent saved-representative replay; no worker math, model or labels."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def logp(vector):
    x = vector.astype(np.float64)
    x = x-float(np.max(x))
    return x-math.log(math.fsum(math.exp(float(v)) for v in x))


def main():
    complete = read('results/r33_COMPLETE.json')
    supervisor = read('logs/r33_SUPERVISOR.json')
    assert complete['status'] == supervisor['status'] == 'complete'
    assert supervisor['exit_code'] == 0 and not supervisor.get('timed_out', False) and not supervisor['overrun']
    for field, path in [('audit_sha256','results/r33_EXECUTION_AUDIT.json'),
                        ('representative_vectors_sha256','results/r33_REPRESENTATIVE_VECTORS.npz'),
                        ('scores_sha256','results/r33_SCORES.json'),
                        ('native_parity_sha256','results/r33_NATIVE_PARITY.json')]:
        assert complete[field] == sha(path)
    audit = read('results/r33_EXECUTION_AUDIT.json')
    first = next(x for x in audit['checks'] if x['type'] == 'first_real_common_gain_math')
    with np.load(ROOT/'results/r33_REPRESENTATIVE_VECTORS.npz', allow_pickle=False) as z:
        def array(ref):
            a = z[ref['key']].copy()
            assert a.dtype == np.float32 and list(a.shape) == ref['shape']
            assert hashlib.sha256(a.tobytes()).hexdigest() == ref['raw_sha256']
            return a
        q1, q4 = (array(first[k]).reshape(-1) for k in ('q1','q4'))
        n1, n4 = (array(first[k]) for k in ('n1','n4'))
        saved = {k:array(ref).reshape(-1) for k, ref in first['outputs'].items()}
    assert q1.shape == q4.shape == (49152,) and n1.shape == n4.shape == (3,49152)
    assert all(np.isfinite(x).all() for x in (q1,q4,n1,n4))
    g = np.float32(.5)*(q4-q1)
    fixed = q4+g
    prior_delta = np.float32(.5)*(n4-n1)
    prior_fixed = n4+prior_delta
    G = logp(fixed)-logp(q4)
    E = np.stack([logp(prior_fixed[j])-logp(n4[j]) for j in range(3)])
    p = np.exp(q4.astype(np.float64)-float(q4.max()))
    p /= math.fsum(float(v) for v in p)
    assert np.isfinite(p).all() and np.all(p > 0)

    def center(vector):
        x = vector-vector[0]
        return x-math.fsum(float(a)*float(b) for a,b in zip(p,x))

    def norm2(vector):
        value = math.fsum(float(w)*float(x)*float(x) for w,x in zip(p,vector))
        assert math.isfinite(value) and value >= 0
        return value

    def common(vectors):
        # Explicit coordinate loop, independent of worker amin/concatenation.
        return np.asarray([min(max(float(row[v]),0.) for row in vectors)
                           -min(max(-float(row[v]),0.) for row in vectors)
                           for v in range(len(q4))], dtype=np.float64)

    c = common([G,*E])
    residual = G-c
    assert np.all(np.where(G >= 0, (residual >= 0)&(residual <= G),
                          (residual >= G)&(residual <= 0)))
    h = center(c)
    main2 = norm2(h)
    zero = bool(np.all(h == 0))
    assert (main2 == 0) == zero
    expected = dict(Fixed=fixed,primary=fixed.copy() if zero else (fixed.astype(np.float64)-h).astype(np.float32))
    diag = dict(main_norm2=main2,h_exact_zero=zero,c_exact_zero=bool(np.all(c == 0)),
                positive_coordinates=int(sum(c > 0)),negative_coordinates=int(sum(c < 0)))

    def matched(direction, tag):
        v = center(direction)
        n2 = norm2(v)
        threshold = 64*np.finfo(np.float64).eps*float(np.max(np.abs(v)))**2
        bad = bool(not zero and n2 <= threshold)
        if zero or bad:
            output = fixed.copy()
            error = 0. if zero else math.sqrt(main2)
        else:
            v = v*math.sqrt(main2/n2)
            output = (fixed.astype(np.float64)-v).astype(np.float32)
            error = abs(math.sqrt(norm2(v))-math.sqrt(main2))
        diag.update({tag+'_unmatched':bad,tag+'_norm_error':error,
                     tag+'_raw_norm2':n2,tag+'_threshold':threshold})
        return output

    generic = np.stack([center(prior_delta[j].astype(np.float64)) for j in range(3)]).mean(axis=0)
    expected['G_matched'] = matched(generic,'G')
    perm = np.random.default_rng(2021).permutation(49152)
    protocol = read('configs/r33_COMMON_GAIN_PILOT_LOCK.json')
    digest = hashlib.sha256(json.dumps(perm.tolist(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert digest == protocol['vocabulary_permutation_sha256']
    expected['vocabulary_null'] = matched(h[perm],'null')
    expected['reference_consensus_matched'] = matched(common(list(E)),'consensus')
    gc = center(g.astype(np.float64))
    g2, r2 = norm2(gc), norm2(gc-h)
    scale_bad = bool(not zero and g2 == 0)
    rho = 1. if zero or scale_bad else math.sqrt(r2/g2)
    expected['matched_scale'] = fixed.copy() if zero or scale_bad else (q4.astype(np.float64)+rho*g.astype(np.float64)).astype(np.float32)
    diag.update(scale_g2=g2,scale_r2=r2,scale_rho=rho,scale_unmatched=scale_bad,
                scale_norm_error=0. if zero or scale_bad else abs(math.sqrt(rho*rho*g2)-math.sqrt(r2)))
    for j in range(3):
        cl = common([G,*[E[i] for i in range(3) if i != j]])
        hl = center(cl)
        n2 = norm2(hl)
        hz = bool(np.all(hl == 0))
        assert (n2 == 0) == hz
        expected['LOO_N'+str(j)] = fixed.copy() if hz else (fixed.astype(np.float64)-hl).astype(np.float32)
        diag['LOO_N'+str(j)+'_norm2'] = n2
    assert saved['Fixed'].tobytes() == fixed.tobytes()
    errors = {k:float(np.max(np.abs(a-saved[k]))) for k,a in expected.items()}
    assert len(errors) == 9 and max(errors.values()) <= 1e-6, errors
    observed = first['diagnostics']
    diag_errors = {}
    for key, value in diag.items():
        if isinstance(value,(bool,int)):
            assert value == observed[key], key
        else:
            diag_errors[key] = abs(value-observed[key])
            assert diag_errors[key] <= 1e-10, (key,diag_errors[key])
    result = dict(status='passed_Root_independent_fullV_representative_math',task=first['task'],
                  vocabulary=49152,independently_reconstructed_outputs=list(expected),
                  max_FP32_output_error=max(errors.values()),FP32_output_errors=errors,
                  max_FP64_diag_error=max(diag_errors.values()),FP64_diag_errors=diag_errors,
                  local_residual_clamp_checked=True,new_model_or_GPU_run=False,labels_opened=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol_sha256=sha('configs/r33_COMMON_GAIN_PILOT_LOCK.json'),
                  representative_vectors_sha256=complete['representative_vectors_sha256'],
                  scope='One fixed saved representative, nine outputs including primary/three LOO/four matched controls/Fixed. No independent all-prefix fullV reconstruction or efficacy claim.')
    with (ROOT/'results/r33_ROOT_MATH_REPLAY.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
