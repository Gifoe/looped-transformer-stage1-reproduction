"""Build all eligible official training records and audit them against reservations."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from dataset_transfer import canonical
from prepare_external import Corpus, normalize_question, science_record
from prepare_pool import gsm, squad
from task_data import encode_target, hash_order, identity, multiple_choice


def sha(path):
    from hashlib import file_digest
    with path.open('rb') as f:
        return file_digest(f,'sha256').hexdigest()


def iter_rows(path):
    if path.suffix=='.parquet':
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    else:
        with path.open(encoding='utf-8') as fio:
            for line in fio:
                yield json.loads(line)


def raw_groups(a):
    return {
        'arc_challenge': (a.transfer/'inputs/upstream_arc_challenge_train.parquet','allenai/ai2_arc/ARC-Challenge'),
        'csqa': (a.transfer/'inputs/csqa_train.jsonl','tau/commonsense_qa@d9021a44'),
        'obqa': (a.transfer/'inputs/obqa_train.parquet','allenai/openbookqa@8c4888010de9b23a2d136ebd31d5f5ea9f3d90fa'),
        'qasc': (a.transfer/'inputs/qasc_train.parquet','allenai/qasc@a34ba204eb9a33b919c10cc08f4f1c8dae5ec070'),
        'scienceqa': (a.external/'inputs/problems.json','lupantech/ScienceQA@2cbf8318e07b9ece895bb2ae605e71e38d623264'),
        'squad': (a.generation/'inputs/squad_train.parquet','rajpurkar/squad@7b6d24c440a36b6815f21b70d25016731768db1f'),
        'gsm8k': (a.generation/'inputs/gsm8k_train.parquet','openai/gsm8k@740312add88f781978c0658806c59bc2815b9866'),
        **{name:(a.reading/f'{name}_train_raw.parquet',name) for name in ('boolq','social_iqa','race')},
    }


def context_hash(value):
    return sha256(identity(value).encode()).hexdigest()


def source(a,name,path):
    if name=='scienceqa':
        problems=json.loads(path.read_text(encoding='utf-8'))
        split=json.loads((a.external/'inputs/pid_splits.json').read_text(encoding='utf-8'))
        for key in split['train']:
            yield dict(problems[key],id=key)
    else:
        yield from iter_rows(path)


def choice(id,stem,context,options,gold,name):
    labels=[chr(65+i) for i in range(len(options))]
    if not 0<=gold<len(labels) or not all(identity(item) for item in options) or len({identity(i) for i in options})!=len(options):
        return None
    prompt=f'Passage: {context}\nQuestion: {stem}' if context else stem
    row=dict(id=str(id),question=prompt,choices={'label':labels,'text':options},answerKey=labels[gold])
    return multiple_choice(row,name,'content')


def convert(raw,name):
    if name=='scienceqa':
        row,failure=science_record(raw['id'],raw)
        if row is None:
            return None,None,None,failure
        cooked=multiple_choice(row['candidate'],name,'content')
        cooked['explanation']=row['explanation']
        return cooked,row['question'],None,None
    if name=='squad':
        return squad(raw),raw['question'],raw['context'],None
    if name=='gsm8k':
        row=gsm(raw,'process')
        row['response']=re.sub(r'<<[^<>]*>>','',row['response'])
        return row,raw['question'],None,None
    if name=='boolq':
        context=raw['passage']; stem=raw['question']
        row=choice(f'{name}/{context_hash(context+chr(10)+stem)}',stem,context,['yes','no'],0 if raw['answer'] else 1,name)
        return row,stem,context,None
    if name=='race':
        context=raw['article']; stem=raw['question']
        id=f"{name}/{raw['example_id']}/{context_hash(stem+str(raw['options']))[:16]}"
        row=choice(id,stem,context,raw['options'],ord(raw['answer'])-65,name)
        return row,stem,context,None
    if name=='social_iqa':
        context=raw['context']; stem=raw['question']
        options=[raw['answer'+letter] for letter in 'ABC']
        id=f'{name}/{context_hash(context+chr(10)+stem+str(options))}'
        row=choice(id,stem,context,options,int(raw['label'])-1,name)
        return row,stem,context,None
    record=canonical(raw,name,name,'train')
    return multiple_choice(record,name,'content'),record['question'],None,None


def reserved_paths(a):
    files=[a.transfer/'inputs'/f'{source}_{split}.parquet' for source in
           ('arc_easy','upstream_arc_challenge','obqa','qasc') for split in ('validation','test')]
    files += [a.transfer/'data'/f'csqa_{split}.jsonl' for split in ('validation','discovery','confirm')]
    files += [a.generation/'inputs'/f'{name}.parquet' for name in ('squad_validation','gsm8k_test')]
    files += sorted(a.advantage.glob('data/*_development.jsonl'))
    files += sorted(a.external.glob('data/panel/dev_*.jsonl'))
    files += sorted(a.external.glob('mixed_full/archive/inputs/*.jsonl'))
    files += [a.feedback/'generation_followup/data/mmlu_pro.parquet']
    return [p for p in files if p.is_file()]


def reservations(a):
    bench=Corpus()
    contexts=set()
    files=reserved_paths(a)
    for path in files:
        if 'squad' in path.stem:
            for row in iter_rows(path):
                if row.get('context'):
                    contexts.add(context_hash(row['context']))
        else:
            for row in iter_rows(path):
                bench.add(normalize_question(row),str(row.get('id',row.get('sample_id',path.name))))
    return bench,contexts,files


def build(a):
    tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    reserved,contexts,documents=reservations(a)
    specs=raw_groups(a)
    reading_lock=a.reading/'READING_INPUTS.json'
    if reading_lock.exists():
        for source_meta in json.loads(reading_lock.read_text()):
            if source_meta['split']=='train':
                name=source_meta['source']
                specs[name]=(specs[name][0],source_meta['url'])
    a.out.mkdir(parents=True,exist_ok=True)
    for name in a.sources:
        path,reference=specs[name]
        result=a.out/f'{name}.jsonl'; manifest=result.with_suffix('.json')
        if manifest.exists():
            info=json.loads(manifest.read_text())
            if info['raw_sha256']!=sha(path) or info['output_sha256']!=sha(result):
                raise ValueError(f'Existing processed source differs: {name}')
            print('REUSED',name, info['retained'],flush=True)
            continue
        excluded_ctx=set(contexts if name=='squad' else [])
        if name in ('race','social_iqa','boolq'):
            validation=a.reading/f'{name}_validation_raw.parquet'
            column={'race':'article','social_iqa':'context','boolq':'passage'}[name]
            excluded_ctx.update(context_hash(r[column]) for r in iter_rows(validation))
        seen_ids,seen_questions=set(),set()
        counts=Counter()
        tmp=result.with_suffix('.partial')
        audit=a.out/f'{name}_excluded.jsonl'
        audit_tmp=audit.with_suffix('.partial')
        with tmp.open('w',encoding='utf-8') as out,audit_tmp.open('w',encoding='utf-8') as log:
            for raw in source(a,name,path):
                counts['input']+=1
                row,stem,context,reason=convert(raw,name)
                duplicate=None
                if row is None:
                    reason=reason or 'invalid_response_or_choices'
                else:
                    domain=context_hash(context) if context else None
                    norm=identity(stem)
                    option_set=tuple(sorted(identity(value) for value in row.get('options',[])))
                    key=(domain,norm,option_set)
                    if domain in excluded_ctx:
                        reason='heldout_context'
                    elif row['sample_id'] in seen_ids or key in seen_questions:
                        reason='duplicate_training_question'
                    elif domain is None and (hit:=reserved.match(norm))[0] is not None:
                        duplicate=hit
                        reason='heldout_near_duplicate'
                    else:
                        try:
                            prefix,whole,n=encode_target(tokenizer,row)
                        except (AssertionError,ValueError):
                            reason='token_boundary'
                        else:
                            if len(whole)>a.max_tokens:
                                reason='over_context_budget'
                            else:
                                row.update(input_tokens=len(whole)-1,response_tokens=n,
                                           fold_key=domain or row['sample_id'],source_question=stem)
                                out.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                                seen_ids.add(row['sample_id']); seen_questions.add(key)
                                counts['retained']+=1
                                continue
                counts[reason]+=1
                log.write(json.dumps(dict(id=(row or raw).get('sample_id',raw.get('id',raw.get('example_id',counts['input']))),
                                          reason=reason,duplicate=duplicate),ensure_ascii=False)+'\n')
        if not counts['retained']:
            raise ValueError(f'Empty eligible source: {name}')
        tmp.replace(result); audit_tmp.replace(audit)
        info=dict(dataset=name,source=reference,raw_sha256=sha(path),output_sha256=sha(result),audit_sha256=sha(audit),
                  split='official train',retained=counts['retained'],counts=dict(counts),maximum_tokens=a.max_tokens,
                  reservation_files={str(p):sha(p) for p in documents},
                  matching='exact context exclusion or word-trigram Jaccard >=0.8 for context-free stems',
                  fold_grouping='question, normalized reading context, or passage')
        manifest.write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
        print('COMPLETE',name,dict(counts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for argument in ('transfer','generation','external','model','feedback','advantage','reading','out'):
        p.add_argument(f'--{argument}',type=Path,required=True)
    p.add_argument('--sources',nargs='+',required=True)
    p.add_argument('--max-tokens',type=int,default=1536)
    build(p.parse_args())
