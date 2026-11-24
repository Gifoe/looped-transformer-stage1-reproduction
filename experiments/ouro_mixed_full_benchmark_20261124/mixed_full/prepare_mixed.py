"""Link immutable archived zero-shot requests; seal one fixed Mixed weight."""
from hashlib import file_digest
import json
import os
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as handle:
        return file_digest(handle,'sha256').hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def link(path, source):
    source=source.resolve(strict=True)
    if path.is_symlink():
        if path.resolve(strict=True)!=source:
            raise ValueError(f'Existing archived link differs at {path}')
        return
    path.symlink_to(source,target_is_directory=source.is_dir())

if __name__=='__main__':
    root=Path('mixed_full')
    stage=Path(os.environ['feedback'])/'stage2'
    followup=stage.parent/'generation_followup'
    archive=root/'archive'
    out=root/'results'
    for folder in (archive/'inputs',archive/'scores',out):
        folder.mkdir(parents=True,exist_ok=True)
    link(archive/'code',stage/'code')
    model=Path(os.environ['model'])
    weight=Path(os.environ.get('OURO_MIXED_WEIGHT',str(Path(os.environ['advantage'])/'results/fits/mixed/controller.pt')))
    expected='ec2e55a64564f608cfeb245da079015db4347a9391072cba7f2949db05185b87'
    if sha(weight)!=expected:
        raise ValueError('Mixed weight does not match the requested SHA-256')
    zero=stage/'zero_shot'
    order={'sciq':1000,'piqa':1838,'arc_challenge':1172,'arc_easy':2376,'winogrande':1267,'mmlu':14042,'hellaswag':10042,'mmlu_pro':12032}
    indexes={base:read(base/'COMPARISON.json') for base in (stage,zero)}
    index={}
    fingerprints={}
    roots={}
    for task,n in order.items():
        if task=='mmlu_pro':
            continue
        base=stage if task in ('sciq','piqa') else zero
        names=[f'{task}.jsonl',f'{task}.jsonl.manifest.json']
        manifest=read(base/'inputs'/names[1])
        if manifest['totals']['documents']!=n or any(x['shots'] for x in manifest['tasks']):
            raise ValueError(f'Wrong zero-shot cohort for {task}')
        for name in names:
            link(archive/'inputs'/name,base/'inputs'/name)
        link(archive/'scores'/task,base/'scores'/task)
        index[task]=indexes[base][task]
        fingerprints[task]=sha(base/'inputs'/names[0])
        roots[task]=base.name
    config=Path('configs/study.json')
    lock=dict(selection='Mixed, fixed by user before scoring',parameters=4096,depth=4,expected=order,
              repository_source='59cc8e5330bd16b6dbf40d687b70ff68e518e3d6',
              arms={'mixed':dict(path=str(weight.resolve()),sha256=expected)},
              model_sha256=sha(model/'model.safetensors'),study_config_sha256=sha(config),primary=read(config)['primary'],
              large_benchmark_scorer_sha256=sha('evaluate_benchmarks.py'),entry_sha256=sha(root/'mixed_evaluate.py'),
              stage2_adapter_sha256=sha(Path(os.environ['unified'])/'seven_unified.py'),
              stage2_sha256=sha(stage/'code/stage2_score.py'),seven_requests_sha256=fingerprints,
              stage2_zero_shot_archive_sources=roots,mmlu_pro_sha256=sha(followup/'data/mmlu_pro.parquet'),
              mmlu_pro_code_sha256=sha(followup/'generation_followup_run.py'),
              mmlu_pro_reference_lock_sha256=sha(followup/'results/mmlu_pro/LOCK.json'))
    for path,value in ((archive/'COMPARISON.json',index),(out/'FREEZE.json',lock)):
        if path.exists():
            if read(path)!=value:
                raise ValueError(f'Frozen inputs changed at {path}')
        else:
            temp=path.with_suffix('.partial')
            temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
            temp.replace(path)
    print('Verified and locked Mixed weight',expected,'at K4 for',sum(order.values()),'questions',flush=True)
