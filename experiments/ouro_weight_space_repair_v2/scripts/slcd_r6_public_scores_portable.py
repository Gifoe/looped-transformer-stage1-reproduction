"""Portable post-package replay of immutable R6 public numeric records.

The canonical ARM producer/verifier remains byte-for-byte in the archive.
Allow at most 1e-13 absolute arithmetic roundoff, zero relative tolerance,
and require every argmax and paired outcome to remain exactly unchanged.
No stored score is modified and no inference coefficient is changed.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np
import slcd_r6_public_scores as base
from slcd_r6_scores import selected_score

ROOT=Path(__file__).resolve().parents[1]
ATOL=1e-13


def main():
    export=base.read('results/slcd_R6_PUBLIC_NUMERIC_EXPORT.json')
    lock=base.read('configs/slcd_R6_EXTERNAL_LOCK.json');audit=base.read('results/slcd_R6_EXTERNAL_ACCEPTANCE.json')
    assert audit['status']=='passed' and audit['scope']=='combined'
    assert export['source_sha256']==base.sha('scripts/slcd_r6_public_scores.py')
    assert export['external_lock_sha256']==audit['external_lock_sha256']==base.sha('configs/slcd_R6_EXTERNAL_LOCK.json')
    assert export['external_acceptance_sha256']==base.sha('results/slcd_R6_EXTERNAL_ACCEPTANCE.json')
    assert audit['final_lock_sha256']==lock['expected_final_sha']==base.sha('configs/slcd_R6_FINAL_LOCK.json')
    assert base.sha('scripts/slcd_r6_scores.py')==lock['ranking_source_sha256']
    tasks={}
    for task,n in base.SIZES.items():
        spec=export['tasks'][task];receipt=base.read(f'results/slcd_r6_{task}_COMPLETE.json')
        assert spec['n']==receipt['n']==n
        assert spec['source_complete_sha256']==audit['tasks'][task]['complete_sha256']==base.sha(f'results/slcd_r6_{task}_COMPLETE.json')
        assert spec['source_score_file_sha256']==audit['tasks'][task]['score_sha256']==receipt['score_file_sha256']
        assert receipt['external_lock_sha256']==audit['external_lock_sha256']
        assert list(base.READOUTS)==spec['readout_order'] and list(base.METHODS)==spec['method_order']
        for p,h in receipt['artifact_sha256'].items():assert base.sha(p)==h==audit['artifact_sha256'][p]
        for p,info in spec['artifacts'].items():assert base.sha(p)==info['sha256'] and (ROOT/p).stat().st_size==info['bytes']
        numeric,meta=base.exported_paths(task)
        with (ROOT/meta).open(encoding='utf8') as stream:ids=[json.loads(line)['id'] for line in stream]
        assert len(ids)==len(set(ids))==n
        with np.load(ROOT/numeric,allow_pickle=False) as archive:d={k:archive[k] for k in archive.files}
        raw=d['original_sum_scores'];pr=d['prior_sum_scores'];lengths=d['choice_char_lengths'];stored=d['ranking_scores']
        assert raw.shape==pr.shape==(n,8,4) and raw.dtype==pr.dtype==np.dtype('float64')
        assert lengths.shape==(n,4) and np.all(lengths>0) and stored.shape==(n,5,4)
        assert d['gold'].shape==(n,) and np.all((d['gold']>=0)&(d['gold']<4))
        original=raw/lengths[:,None,:];prior=pr/lengths[:,None,:]
        pred=np.empty((n,5),dtype=np.int16);max_error=0.;roundoff_rows=0
        for i in range(n):
            m={name:dict(normalized_scores=original[i,j].tolist()) for j,name in enumerate(base.READOUTS)}
            pm={name:dict(normalized_scores=prior[i,j].tolist()) for j,name in enumerate(base.READOUTS)}
            f=original[i,6];k4=original[i,3];a1=f+(k4-original[i,4]);g=f-k4;e=prior[i,6]-prior[i,3]
            gc=g-g.mean();ec=e-e.mean();den=float(np.dot(ec,ec));num=float(np.dot(gc,ec))
            alpha=0. if den==0 or num<=0 else 1. if num>=den else num/den
            independent=a1.copy() if alpha==0 else a1-alpha*e
            selected=selected_score(m,pm,'r6_prior_projection')
            assert np.array_equal(selected,independent),'Independent local formula mismatch'
            expected=np.stack([k4,f,original[i,7],a1,selected])
            assert np.isfinite(expected).all() and np.isfinite(stored[i]).all()
            error=float(np.max(np.abs(expected-stored[i])));max_error=max(max_error,error)
            roundoff_rows+=not np.array_equal(expected,stored[i])
            assert np.allclose(expected,stored[i],atol=ATOL,rtol=0),'Roundoff exceeds declared absolute tolerance'
            pred[i]=expected.argmax(axis=1)
            assert np.array_equal(pred[i],stored[i].argmax(axis=1)),'Any changed argmax is rejected'
        rng=np.random.default_rng(2026);comparisons=[]
        for j,name in enumerate(base.METHODS):
            good=pred[:,j]==d['gold']
            for baseline in ['fixed','A1_g1p0']:
                row=dict(method=name,baseline=baseline,**base.pair_stats(good,pred[:,base.METHODS.index(baseline)]==d['gold'],rng));comparisons.append(row)
                old=next(r for r in receipt['paired_comparisons'] if r['method']==name and r['baseline']==baseline)
                for k,v in row.items():
                    if isinstance(v,float):assert math.isclose(v,old[k],rel_tol=1e-13,abs_tol=1e-13),(task,name,baseline,k)
                    else:assert v==old[k]
        lookup={qid:i for i,qid in enumerate(ids)};seen=set()
        with (ROOT/f'results/slcd_r6_{task}_full_scores.csv').open(newline='',encoding='utf8') as stream:
            for row in csv.DictReader(stream):
                i=lookup[row['id']];j=base.METHODS.index(row['method']);choice=int(row['choice']);key=(i,j,choice)
                assert key not in seen and 0<=choice<4;seen.add(key)
                assert float(row['normalized_score'])==stored[i,j,choice],'CSV is compared to exact saved values, not rounded replay'
                assert int(row['prediction'])==pred[i,j] and int(row['gold'])==d['gold'][i]
        assert len(seen)==n*20
        tasks[task]=dict(status='passed',n=n,max_absolute_arithmetic_error=max_error,non_bitwise_equal_questions=roundoff_rows,
                        prediction_differences=0,full_candidate_csv_rows=len(seen),raw_LL_values=n*8*4*2,
                        correct={name:int((pred[:,j]==d['gold']).sum()) for j,name in enumerate(base.METHODS)},paired_statistics=comparisons,
                        source_score_file_sha256=spec['source_score_file_sha256'])
    result=dict(status='passed',scope='Portable numeric replay; canonical serialized scores and all hashes remain exact',
                absolute_arithmetic_tolerance=ATOL,relative_arithmetic_tolerance=0.,argmax_tolerance=0,
                tasks=tasks,export_receipt_sha256=base.sha('results/slcd_R6_PUBLIC_NUMERIC_EXPORT.json'),
                canonical_producer_source_sha256=base.sha('scripts/slcd_r6_public_scores.py'),
                frozen_ranker_source_sha256=base.sha('scripts/slcd_r6_scores.py'),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numpy_version=np.__version__,
                model_forwards=0,stored_scores_modified=False,
                limits='This arithmetic tolerance is only an audit portability rule. It is not an inference parameter, new method, calibrated probability or a model-likelihood regeneration.')
    (ROOT/'results/slcd_R6_PORTABLE_NUMERIC_ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='passed',tasks={t:{k:r[k] for k in ['n','max_absolute_arithmetic_error','prediction_differences','correct']} for t,r in tasks.items()}),indent=2))


if __name__=='__main__':main()
