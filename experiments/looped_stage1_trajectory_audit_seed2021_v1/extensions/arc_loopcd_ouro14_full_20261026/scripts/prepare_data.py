"""New disjoint manifests and requests rendered by the actual pinned harness."""
import hashlib
import json
import random
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
from common import ROOT,BASE,CFG,save,read
from datasets import Dataset,DatasetDict
from lm_eval.api.task import ConfigurableTask

sys.path.insert(0,str(BASE/'scripts'))
from prepare_resources import download
sources=json.loads((BASE/'data/manifest_audit.json').read_text())['sources']
snapshot_records=[]

def clean(value):
    if isinstance(value,dict):return {key:clean(v) for key,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value

def table(repo,folder,split):
    metadata=sources[repo];revision=metadata['sha']
    names=sorted(x['rfilename'] for x in metadata['siblings'] if x['rfilename'].startswith(folder+'/') and '/'+split+'-' in x['rfilename'] and x['rfilename'].endswith('.parquet'))
    assert names,(repo,folder,split)
    frames=[]
    for name in names:
        prior=BASE/'data/snapshots'/repo.replace('/','--')/name
        dest=prior if prior.exists() else ROOT/'data/snapshots'/repo.replace('/','--')/name
        if not dest.exists():
            result=download(f'https://hf-mirror.com/datasets/{repo}/resolve/{revision}/{name}?download=true',dest)
            assert 'error' not in result,result
        with dest.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        snapshot_records.append(dict(repo=repo,revision=revision,file=name,path=str(dest),sha256=digest))
        frames.append(pd.read_parquet(dest))
    return [clean(row.to_dict()) for _,row in pd.concat(frames,ignore_index=True).iterrows()],revision

def normalize_question(text):return ' '.join(text.casefold().split())

def record(task,split,index,doc,revision):
    if task=='arc_challenge':question=doc['question'];choices=doc['choices']['text'];gold=doc['choices']['label'].index(str(doc['answerKey']));uid=doc['id']
    elif task=='openbookqa':question=doc['question_stem'];choices=doc['choices']['text'];gold=doc['choices']['label'].index(str(doc['answerKey']));uid=doc['id']
    else:question=doc['question'];choices=doc['choices'];gold=doc['answer'];uid=index
    return dict(id=f'{task}/{split}/{uid}',task=task,official_split=split,source_index=index,question=question,choices=choices,gold=gold,source_doc=doc,dataset_revision=revision)

def sample(rows,n,tag,excluded_ids,excluded_questions):
    pool=[row for row in rows if row['id'] not in excluded_ids and normalize_question(row['question']) not in excluded_questions]
    random.Random(2021+int(hashlib.sha256(('extension20261025/'+tag).encode()).hexdigest()[:8],16)).shuffle(pool)
    result=[];seen=set()
    for row in pool:
        q=normalize_question(row['question'])
        if q not in seen:result.append(row);seen.add(q)
        if len(result)==n:break
    assert len(result)==n,(tag,len(result),n)
    return result

def task_engine(task,train,dev,split_name,eval_docs):
    if task=='arc_challenge':
        config=yaml.safe_load((ROOT/'sources/harness_arc_easy.yaml').read_text());config.update(task=task,dataset_name='ARC-Challenge')
    elif task=='openbookqa':config=yaml.safe_load((ROOT/'sources/harness_openbook.yaml').read_text())
    else:
        config=yaml.safe_load((ROOT/'sources/harness_mmlu.yaml').read_text());subject=task.removeprefix('mmlu_')
        config.update(task=task,dataset_name=subject,description='The following are multiple choice questions (with answers) about '+subject.replace('_',' ')+'.\n\n')
    datasets=DatasetDict({split_name:Dataset.from_list(eval_docs)})
    if train:datasets['train']=Dataset.from_list(train)
    if dev:datasets['dev']=Dataset.from_list(dev)
    config['custom_dataset']=lambda **kwargs:datasets
    config['test_split']=split_name;config['validation_split']=None
    config['dataset_kwargs']={};config['num_fewshot']=CFG['fewshot']['mmlu' if task.startswith('mmlu_') else task]
    engine=ConfigurableTask(config=config)
    engine.set_fewshot_seed(2021)
    return engine

def requests_for(rows,train,dev,split_name):
    task=rows[0]['task'];engine=task_engine(task,train,dev,split_name,[row['source_doc'] for row in rows]);out=[]
    metric=CFG['mc_primary_metric']['mmlu' if task.startswith('mmlu_') else task]
    for row in rows:
        doc=row['source_doc'];context=engine.fewshot_context(doc,engine.config.num_fewshot)
        requests=engine.construct_requests(doc,context,metadata=(task,row['source_index'],1))
        arguments=[list(request.args) for request in requests]
        choices=engine.doc_to_choice(doc)
        assert all(pair[0]==context for pair in arguments)
        assert int(engine.doc_to_target(doc))==row['gold']
        out.append(dict(**row,prompt=context,request_pairs=arguments,choice_char_lengths=[len(c) for c in choices],fewshot_count=engine.config.num_fewshot,primary_metric=metric,prompt_sha256=hashlib.sha256(context.encode()).hexdigest()))
    return out

def write_manifest(name,rows):
    payload=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows);dest=ROOT/'data'/(name+'.jsonl')
    if dest.exists():assert dest.read_text(encoding='utf-8')==payload,('Frozen manifest would change',name)
    dest.write_text(payload,encoding='utf-8',newline='\n')
    return dict(n=len(rows),sha256=hashlib.sha256(payload.encode()).hexdigest())

def main():
    old=read(BASE/'data/validation.jsonl')+read(BASE/'data/qwen_test.jsonl')
    excluded_ids={r['id'] for r in old};excluded_questions={normalize_question(r['question']) for r in old}
    manifests={'validation':[],'qwen_test':[],'ouro_test':[]}
    for task,repo,folder in [('arc_challenge','allenai/ai2_arc','ARC-Challenge'),('openbookqa','allenai/openbookqa','main')]:
        train,revision=table(repo,folder,'train')
        for split,count in [('validation',100),('test',200)]:
            docs,revision=table(repo,folder,split)
            rows=[record(task,split,index,doc,revision) for index,doc in enumerate(docs)]
            selected=sample(rows,count,task+'/'+split,excluded_ids,excluded_questions)
            excluded_ids.update(r['id'] for r in selected);excluded_questions.update(normalize_question(r['question']) for r in selected)
            rendered=requests_for(selected,train,None,split)
            manifests['validation' if split=='validation' else 'qwen_test']+=rendered
            if split=='test':manifests['ouro_test']+=rendered[:100]
    for subject in CFG['mmlu_subjects']:
        task='mmlu_'+subject;dev,_=table('cais/mmlu',subject,'dev')
        for split,n in [('validation',10),('test',40)]:
            docs,revision=table('cais/mmlu',subject,split)
            rows=[record(task,split,i,doc,revision) for i,doc in enumerate(docs)]
            selected=sample(rows,n,subject+'/'+split,excluded_ids,excluded_questions)
            excluded_ids.update(r['id'] for r in selected);excluded_questions.update(normalize_question(r['question']) for r in selected)
            rendered=requests_for(selected,None,dev,split)
            manifests['validation' if split=='validation' else 'qwen_test']+=rendered
            if split=='test':manifests['ouro_test']+=rendered[:20]
    demonstrations=read(BASE/'data/math_fewshot.jsonl');old_math=read(BASE/'data/math_test.jsonl')
    manifests['math_fewshot']=demonstrations
    for split,count,name in [('train',CFG['math']['validation_count'],'math_validation'),('test',CFG['math']['test_count'],'math_test')]:
        docs,revision=table('openai/gsm8k','main',split)
        rows=[dict(id=f'gsm8k/{split}/{i}',task='gsm8k',official_split=split,source_index=i,question=doc['question'],answer=doc['answer'],gold=doc['answer'].split('####')[-1].strip(),dataset_revision=revision) for i,doc in enumerate(docs)]
        manifests[name]=sample(rows,count,'gsm8k/'+split,{r['id'] for r in demonstrations+old_math},{normalize_question(r['question']) for r in demonstrations+old_math})
    audit={name:write_manifest(name,rows) for name,rows in manifests.items()}
    ids={name:{r['id'] for r in rows} for name,rows in manifests.items()}
    assert not ids['validation']&ids['qwen_test'] and ids['ouro_test']<=ids['qwen_test']
    assert not (ids['validation']|ids['qwen_test'])&{r['id'] for r in old}
    assert not ids['math_validation']&ids['math_test']
    save('data/manifest_audit.json',dict(seed=2021,manifests=audit,snapshots=snapshot_records,harness_version='0.4.9',prior_stage1_ids_excluded=True,exact_question_duplicates_excluded=True))
    print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
