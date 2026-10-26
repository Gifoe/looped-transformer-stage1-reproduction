"""Immutable disjoint manifests, pinned public dataset snapshots, seed 2021."""
import hashlib
import json
import random
from pathlib import Path
import requests
import pandas as pd
from prepare_resources import download

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
sess=requests.Session(); sess.trust_env=False
sources={}

def table(repo,pattern):
    if repo not in sources:
        r=sess.get('https://hf-mirror.com/api/datasets/'+repo,timeout=30);r.raise_for_status()
        sources[repo]=r.json()
    meta=sources[repo];rev=meta['sha']
    names=[x['rfilename'] for x in meta['siblings'] if pattern(x['rfilename'])]
    if not names: raise RuntimeError(f'No files found: {repo}, {[x["rfilename"] for x in meta["siblings"]]}')
    frames=[]
    for name in sorted(names):
        dest=DATA/'snapshots'/repo.replace('/','--')/name
        item=download(f'https://hf-mirror.com/datasets/{repo}/resolve/{rev}/{name}?download=true',dest)
        if 'error' in item: raise RuntimeError(item)
        frames.append(pd.read_parquet(dest))
    return pd.concat(frames,ignore_index=True),rev,names

def take(rows,n,tag):
    rng=random.Random(2021+int(hashlib.sha256(tag.encode()).hexdigest()[:8],16))
    indices=list(range(len(rows)));rng.shuffle(indices)
    return [rows[i] for i in indices[:n]]

def mc_row(task,split,index,row,rev):
    if task=='arc_challenge':
        options=list(row['choices']['text']); labels=list(row['choices']['label'])
        gold=labels.index(str(row['answerKey'])); question=row['question']; uid=str(row['id'])
    elif task=='openbookqa':
        options=list(row['choices']['text']);labels=list(row['choices']['label'])
        gold=labels.index(row['answerKey']);question=row['question_stem'];uid=str(row['id'])
    else:
        options=list(row['choices']);gold=int(row['answer']);question=row['question'];uid=str(index)
    return dict(id=f'{task}/{split}/{uid}',task=task,official_split=split,question=question,
                choices=options,gold=gold,dataset_revision=rev)

def main():
    config=json.loads((ROOT/'configs/protocol.json').read_text(encoding='utf-8-sig'))
    manifests={'validation':[],'qwen_test':[],'ouro_test':[],'math_test':[],'math_fewshot':[]}
    for task,repo,folder in [('arc_challenge','allenai/ai2_arc','ARC-Challenge/'),('openbookqa','allenai/openbookqa','main/')]:
        for split,n in [('validation',50),('test',200)]:
            df,rev,_=table(repo,lambda name,sp=split,fo=folder:name.startswith(fo) and '/'+sp+'-' in name and name.endswith('.parquet'))
            rows=[mc_row(task,split,i,row,rev) for i,row in df.iterrows()]
            sample=take(rows,n,task+'/'+split)
            if split=='validation':manifests['validation']+=sample
            else:
                manifests['qwen_test']+=sample
                manifests['ouro_test']+=sample[:100]
    for subject in config['mmlu_subjects']:
        df,rev,_=table('cais/mmlu',lambda name,sub=subject:name.startswith(sub+'/') and '/test-' in name and name.endswith('.parquet'))
        rows=[mc_row('mmlu_'+subject,'test',i,row,rev) for i,row in df.iterrows()]
        selected=take(rows,40,subject)
        manifests['qwen_test']+=selected;manifests['ouro_test']+=selected[:20]
    for split,n in [('train',3),('test',100)]:
        df,rev,_=table('openai/gsm8k',lambda name,sp=split:name.startswith('main/') and '/'+sp+'-' in name and name.endswith('.parquet'))
        rows=[dict(id=f'gsm8k/{split}/{i}',task='gsm8k',official_split=split,question=row['question'],answer=row['answer'],gold=row['answer'].split('####')[-1].strip(),dataset_revision=rev) for i,row in df.iterrows()]
        manifests['math_fewshot' if split=='train' else 'math_test']=take(rows,n,'gsm8k/'+split)
    ids={k:{r['id'] for r in v} for k,v in manifests.items()}
    assert not ids['validation'] & ids['qwen_test']
    assert not ids['validation'] & ids['ouro_test']
    assert ids['ouro_test'] <= ids['qwen_test']
    records={}
    for name,rows in manifests.items():
        payload=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows)
        dest=DATA/(name+'.jsonl')
        if dest.exists() and dest.read_text(encoding='utf-8')!=payload:
            raise RuntimeError(f'Existing manifest would change: {dest}')
        dest.write_text(payload,encoding='utf-8')
        records[name]=dict(n=len(rows),sha256=hashlib.sha256(payload.encode()).hexdigest())
    report=dict(seed=2021,manifests=records,sources=sources)
    (DATA/'manifest_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(records),flush=True)

if __name__=='__main__':main()
