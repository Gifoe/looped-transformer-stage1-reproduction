"""Lossless numeric external-score release, without source text or token IDs.

export: server only, after full independent acceptance; reads complete JSONLs.
verify: public records only; recomputes every frozen ranking and paired statistic.
Neither command runs a model or selects a method. No pickle/object NumPy arrays.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from slcd_r6_scores import selected_score

READOUTS=tuple(f'native_k{k}' for k in range(1,7))+('loopcd_k4_fixed','loopcd_k4_adaptive')
METHODS=('native4','fixed','adaptive','A1_g1p0','r6_prior_projection')
SIZES={'hellaswag':10042,'openbookqa':500}


def sha(path):
    h=hashlib.sha256()
    with (ROOT/path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads((ROOT/path).read_text(encoding='utf8'))


def write(path,value):
    (ROOT/path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def exported_paths(task):
    return (f'artifacts/slcd_r6_{task}_numeric.npz',f'artifacts/slcd_r6_{task}_input_hashes.jsonl')


def export():
    audit=read('results/slcd_R6_EXTERNAL_ACCEPTANCE.json')
    assert audit['status']=='passed' and audit['scope']=='combined'
    lock=read('configs/slcd_R6_EXTERNAL_LOCK.json');tasks={}
    assert audit['external_lock_sha256']==sha('configs/slcd_R6_EXTERNAL_LOCK.json')
    assert audit['final_lock_sha256']==lock['expected_final_sha']==sha('configs/slcd_R6_FINAL_LOCK.json')
    assert sha('scripts/slcd_r6_scores.py')==lock['ranking_source_sha256']
    for task,n in SIZES.items():
        receipt_path=f'results/slcd_r6_{task}_COMPLETE.json';receipt=read(receipt_path)
        assert receipt['status']=='complete' and receipt['n']==n
        assert sha(receipt['score_file'])==receipt['score_file_sha256']
        assert sha(receipt_path)==audit['tasks'][task]['complete_sha256']
        assert receipt['score_file_sha256']==audit['tasks'][task]['score_sha256']
        assert receipt['external_lock_sha256']==audit['external_lock_sha256']
        for p,value in receipt['artifact_sha256'].items():assert sha(p)==value==audit['artifact_sha256'][p]
        original=np.empty((n,8,4),dtype=np.float64);prior=np.empty_like(original)
        ranking=np.empty((n,5,4),dtype=np.float64)
        gold=np.empty(n,dtype=np.int16);lengths=np.empty((n,4),dtype=np.int32)
        main_tokens=np.empty_like(lengths);prior_tokens=np.empty_like(lengths)
        equal_tokens=np.empty((n,4),dtype=np.bool_);origins=np.empty(n,dtype=np.bool_)
        ids=set();metadata=[]
        with (ROOT/receipt['score_file']).open(encoding='utf8') as f:
            for i,line in enumerate(f):
                row=json.loads(line);assert i<n and row['id'] not in ids;ids.add(row['id'])
                assert len(row['choice_char_lengths'])==4
                gold[i]=row['gold'];lengths[i]=row['choice_char_lengths']
                main_tokens[i]=row['choice_token_counts'];prior_tokens[i]=row['prior_token_counts']
                equal_tokens[i]=row['target_ids_equal_by_choice'];origins[i]=row['main_origin']=='old_authority'
                for j,name in enumerate(READOUTS):
                    original[i,j]=row['original_methods'][name]['sum_scores']
                    prior[i,j]=row['prior_methods'][name]['sum_scores']
                    assert np.array_equal(original[i,j]/lengths[i],row['original_methods'][name]['normalized_scores'])
                    assert np.array_equal(prior[i,j]/lengths[i],row['prior_methods'][name]['normalized_scores'])
                for j,name in enumerate(METHODS):ranking[i,j]=row['methods'][name]['normalized_scores']
                metadata.append(dict(id=row['id'],prompt_sha256=row['prompt_sha256'],
                    ordered_request_pairs_sha256=object_sha(row['request_pairs']),
                    ordered_continuation_sha256=[hashlib.sha256(p[1].encode()).hexdigest() for p in row['request_pairs']],
                    prior_cache_references=[{k:v for k,v in ref.items() if k in ['origin','cache_key','entry_sha256','input_key']} for ref in row['prior_cache_entries']],
                    signature=row['signature']))
        assert len(ids)==len(metadata)==n
        numeric,meta=exported_paths(task)
        np.savez_compressed(ROOT/numeric,original_sum_scores=original,prior_sum_scores=prior,
                            ranking_scores=ranking,gold=gold,choice_char_lengths=lengths,
                            main_target_token_counts=main_tokens,prior_target_token_counts=prior_tokens,
                            target_ids_equal_by_choice=equal_tokens,old_main_origin=origins)
        with (ROOT/meta).open('w',encoding='utf8') as f:
            for row in metadata:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        tasks[task]=dict(n=n,readout_order=READOUTS,method_order=METHODS,source_score_file=receipt['score_file'],
                        source_score_file_sha256=receipt['score_file_sha256'],source_complete_sha256=sha(receipt_path),
                        artifacts={p:dict(bytes=(ROOT/p).stat().st_size,sha256=sha(p)) for p in [numeric,meta]},
                        values='All sixteen raw complete-candidate LL readouts as exact FP64, all five final ranking vectors, labels, lengths, counts, origin and input hashes.',
                        omitted='Source text, raw parquet, reversible token IDs, and unused descriptive per-row timings. Full raw JSONL/cache is preserved on the experiment server.')
    write('results/slcd_R6_PUBLIC_NUMERIC_EXPORT.json',dict(status='exported_after_independent_full_acceptance',tasks=tasks,
            external_acceptance_sha256=sha('results/slcd_R6_EXTERNAL_ACCEPTANCE.json'),external_lock_sha256=sha('configs/slcd_R6_EXTERNAL_LOCK.json'),
            source_sha256=sha('scripts/slcd_r6_public_scores.py'),model_forwards=0,lossless_scope='Numeric likelihoods/ranking vectors, not a byte-for-byte raw JSONL copy'))
    print(json.dumps({task:{p:v['bytes'] for p,v in spec['artifacts'].items()} for task,spec in tasks.items()},indent=2))


def pair_stats(good,base,rng):
    from scipy.stats import binomtest
    n=len(good);rescues=int((good&~base).sum());damage=int((base&~good).sum())
    draws=rng.multinomial(n,[rescues/n,damage/n,1-(rescues+damage)/n],size=50000)
    delta=100*(draws[:,0]-draws[:,1])/n
    return dict(n=n,correct=int(good.sum()),accuracy=float(good.mean()),corrections=rescues,regressions=damage,
                net_correct=rescues-damage,delta_pp=100*(rescues-damage)/n,
                exact_mcnemar_p=1. if rescues+damage==0 else float(binomtest(rescues,rescues+damage,.5).pvalue),
                paired_bootstrap_low_pp=float(np.quantile(delta,.025)),paired_bootstrap_high_pp=float(np.quantile(delta,.975)))


def verify():
    export_receipt=read('results/slcd_R6_PUBLIC_NUMERIC_EXPORT.json');lock=read('configs/slcd_R6_EXTERNAL_LOCK.json')
    audit=read('results/slcd_R6_EXTERNAL_ACCEPTANCE.json');assert audit['status']=='passed' and audit['scope']=='combined'
    assert export_receipt['source_sha256']==sha('scripts/slcd_r6_public_scores.py')
    assert export_receipt['external_lock_sha256']==sha('configs/slcd_R6_EXTERNAL_LOCK.json')
    assert export_receipt['external_lock_sha256']==audit['external_lock_sha256']
    assert audit['final_lock_sha256']==lock['expected_final_sha']==sha('configs/slcd_R6_FINAL_LOCK.json')
    assert export_receipt['external_acceptance_sha256']==sha('results/slcd_R6_EXTERNAL_ACCEPTANCE.json')
    assert sha('scripts/slcd_r6_scores.py')==lock['ranking_source_sha256']
    results={}
    for task,n in SIZES.items():
        spec=export_receipt['tasks'][task];receipt=read(f'results/slcd_r6_{task}_COMPLETE.json')
        assert spec['n']==receipt['n']==n and receipt['score_file_sha256']==spec['source_score_file_sha256']
        assert spec['source_complete_sha256']==sha(f'results/slcd_r6_{task}_COMPLETE.json')
        assert spec['source_complete_sha256']==audit['tasks'][task]['complete_sha256']
        assert spec['source_score_file_sha256']==audit['tasks'][task]['score_sha256']
        assert receipt['external_lock_sha256']==audit['external_lock_sha256']
        for p,value in receipt['artifact_sha256'].items():assert sha(p)==value==audit['artifact_sha256'][p]
        assert list(READOUTS)==spec['readout_order'] and list(METHODS)==spec['method_order']
        for p,meta in spec['artifacts'].items():assert sha(p)==meta['sha256'] and (ROOT/p).stat().st_size==meta['bytes']
        numeric,meta=exported_paths(task)
        with (ROOT/meta).open(encoding='utf8') as f:identities=[json.loads(line) for line in f]
        ids=[r['id'] for r in identities];assert len(ids)==len(set(ids))==n
        with np.load(ROOT/numeric,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
        raw=arrays['original_sum_scores'];prior_raw=arrays['prior_sum_scores'];lengths=arrays['choice_char_lengths']
        assert raw.shape==prior_raw.shape==(n,8,4) and raw.dtype==prior_raw.dtype==np.dtype('float64')
        assert lengths.shape==(n,4) and np.all(lengths>0)
        assert arrays['ranking_scores'].shape==(n,5,4) and arrays['gold'].shape==(n,)
        assert np.all((arrays['gold']>=0)&(arrays['gold']<4))
        original=raw/lengths[:,None,:];prior=prior_raw/lengths[:,None,:]
        predictions=np.empty((n,5),dtype=np.int16)
        for i in range(n):
            main={name:dict(normalized_scores=original[i,j].tolist()) for j,name in enumerate(READOUTS)}
            pm={name:dict(normalized_scores=prior[i,j].tolist()) for j,name in enumerate(READOUTS)}
            fixed=original[i,6];k4=original[i,3];a1=fixed+(k4-original[i,4])
            guidance=fixed-k4;e=prior[i,6]-prior[i,3];gc=guidance-guidance.mean();ec=e-e.mean()
            den=float(np.dot(ec,ec));num=float(np.dot(gc,ec));alpha=0. if den==0 or num<=0 else 1. if num>=den else num/den
            independent=a1.copy() if alpha==0 else a1-alpha*e
            selected=selected_score(main,pm,'r6_prior_projection');assert np.array_equal(selected,independent)
            expected=np.stack([k4,fixed,original[i,7],a1,selected])
            assert np.isfinite(expected).all() and np.array_equal(expected,arrays['ranking_scores'][i])
            predictions[i]=expected.argmax(axis=1)
        rng=np.random.default_rng(2026);stats=[]
        for j,name in enumerate(METHODS):
            good=predictions[:,j]==arrays['gold']
            for base in ['fixed','A1_g1p0']:
                row=dict(method=name,baseline=base,**pair_stats(good,predictions[:,METHODS.index(base)]==arrays['gold'],rng));stats.append(row)
                old=next(r for r in receipt['paired_comparisons'] if r['method']==name and r['baseline']==base)
                for k,v in row.items():
                    if isinstance(v,float):assert math.isclose(v,old[k],rel_tol=1e-13,abs_tol=1e-13),(task,name,base,k,v,old[k])
                    else:assert old[k]==v
        # Independent numeric comparison with every public final candidate CSV row.
        seen=set();index={qid:i for i,qid in enumerate(ids)}
        with (ROOT/f'results/slcd_r6_{task}_full_scores.csv').open(newline='',encoding='utf8') as f:
            for row in csv.DictReader(f):
                i=index[row['id']];j=METHODS.index(row['method']);choice=int(row['choice']);key=(i,j,choice)
                assert key not in seen and 0<=choice<4;seen.add(key)
                assert float(row['normalized_score'])==arrays['ranking_scores'][i,j,choice]
                assert int(row['prediction'])==predictions[i,j] and int(row['gold'])==arrays['gold'][i]
        assert len(seen)==n*20
        results[task]=dict(status='passed',n=n,candidate_score_rows_checked=n*20,
                           raw_likelihood_values_checked=n*8*4*2,rankings_checked=n*5,
                           correct={name:int((predictions[:,j]==arrays['gold']).sum()) for j,name in enumerate(METHODS)},
                           paired_statistics=stats,source_score_file_sha256=spec['source_score_file_sha256'],numeric_artifact_sha256={p:v['sha256'] for p,v in spec['artifacts'].items()})
    output=dict(status='passed',scope='public lossless numeric records; no source text or token IDs needed',tasks=results,
                export_receipt_sha256=sha('results/slcd_R6_PUBLIC_NUMERIC_EXPORT.json'),source_sha256=sha('scripts/slcd_r6_public_scores.py'),
                limits=['Does not independently regenerate model likelihoods or original input text/tokenization. Server full audit covers exact inputs and cache keys.',
                        'No new method, tuning, calibrated probability, causal mechanism or runtime speed claim.'])
    write('results/slcd_R6_PUBLIC_NUMERIC_ACCEPTANCE.json',output)
    print(json.dumps(dict(status='passed',correct={t:r['correct'] for t,r in results.items()}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['export','verify']);args=parser.parse_args()
    {'export':export,'verify':verify}[args.command]()
