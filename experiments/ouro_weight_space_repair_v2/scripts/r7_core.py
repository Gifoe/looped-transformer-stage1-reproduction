"""R7 paths, stage gates, exact complete-candidate baseline reconstruction.

Development reads discovery384/selection192 only. The first576 ordered R5
prior rows are consumed without parsing confirmation rows. No test outcomes
are opened here. All writes use the new r7_ namespace in existing parents.
"""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE_COMMIT='81d1c83b79461bf69882a0ffd4cf39b38dc3ba36'
BASE_METHODS=('native4','fixed','adaptive','A1','M0_R6')
SIZES={'discovery':384,'selection':192,'confirmation':517,'validation':299,'test':1172}


def sha(relative):
    p=ROOT/relative;h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(relative):return json.loads((ROOT/relative).read_text(encoding='utf8'))


def save(relative,value):
    p=ROOT/relative
    assert p.name.startswith('r7_') and p.parent.is_dir() and p.resolve().is_relative_to(ROOT.resolve())
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)


def csvout(relative,values):
    p=ROOT/relative;assert p.name.startswith('r7_') and p.parent.is_dir()
    fields=list(dict.fromkeys(key for row in values for key in row))
    with p.open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(values)


def jsonlines(relative):
    with (ROOT/relative).open(encoding='utf8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def gated_rows(split):
    if split in ['confirmation','validation']:
        lock=read('configs/r7_CANDIDATE_LOCK.json');assert lock['status']=='frozen_before_confirmation'
    elif split=='test':
        lock=read('configs/r7_FINAL_METHOD_LOCK.json');assert lock['test_access_authorized'] and lock['primary']!='M0_R6'
    else:assert split in ['discovery','selection']
    if split in ['confirmation','validation','test']:
        assert sha('data/slcd_'+split+'.jsonl')==lock['dataset_sha256'][split], 'Locked held-out input changed before parsing'
    else:
        historical=read('configs/slcd_R2_DEVELOPMENT_LOCK.json')
        assert sha('data/slcd_'+split+'.jsonl')==historical['train_data_sha256'][split]
    values=list(jsonlines('data/slcd_'+split+'.jsonl'))
    assert len(values)==len({v['id'] for v in values})==SIZES[split]
    return values


def baseline_vectors(original,prior):
    from slcd_r6_scores import selected_score
    vec=lambda key:np.asarray(original[key]['normalized_scores'],dtype=np.float64)
    f=vec('loopcd_k4_fixed');q4=vec('native_k4');q5=vec('native_k5')
    result=dict(native4=q4,fixed=f,adaptive=vec('loopcd_k4_adaptive'),A1=f+(q4-q5),
                M0_R6=selected_score(original,prior,'r6_prior_projection'))
    assert all(v.ndim==1 and v.shape==f.shape and np.isfinite(v).all() for v in result.values())
    return result


def development_baselines():
    manifests={s:gated_rows(s) for s in ['discovery','selection']}
    ordered=[row for s in manifests for row in manifests[s]]
    n0path='results/slcd_r5_priors_train.jsonl';prior_receipt=read('results/slcd_r5_priors_train_COMPLETE.json')
    assert prior_receipt['status']=='complete' and sha(n0path)==prior_receipt['score_file_sha256']
    # Deliberately stop after576 rather than filtering/decoding all1093 labels.
    with (ROOT/n0path).open(encoding='utf8') as stream:
        priors=[json.loads(line) for line in itertools.islice(stream,len(ordered))]
    assert [x['id'] for x in priors]==[x['id'] for x in ordered], 'Prior file must have the authorized development prefix'
    prior={p['id']:p for p in priors};answer=[]
    for split,data in manifests.items():
        path=f'results/slcd_r2_{split}_development.jsonl';receipt=read(path.replace('.jsonl','_COMPLETE.json'))
        assert sha(path)==receipt['score_file_sha256'] and receipt['status']=='complete'
        original={r['id']:r for r in jsonlines(path)};assert set(original)=={r['id'] for r in data}
        for item in data:
            row=original[item['id']];p=prior[item['id']]
            assert row['gold']==p['gold']==item['gold']
            assert row['prompt_sha256']==p['prompt_sha256']==item['prompt_sha256']
            assert row['choice_char_lengths']==p['choice_char_lengths']==item['choice_char_lengths']
            for methods in [row['original_methods'],p['prior_methods']]:
                for value in methods.values():
                    assert np.array_equal(np.asarray(value['sum_scores'])/np.asarray(item['choice_char_lengths']),value['normalized_scores'])
            answer.append(dict(split=split,item=item,original=row['original_methods'],prior=p['prior_methods'],
                               baseline=baseline_vectors(row['original_methods'],p['prior_methods'])))
    return answer
