"""Fresh semantic evidence pilot, no held-out/external access without own gate."""
import json
from r7_core import ROOT,read,sha,digest,jsonlines

SOURCES=('r10_core.py','r10_scores.py','r10_worker.py','r9_scores.py','r8_scores.py','r9_core.py','r9_worker.py',
         'r7_core.py','r7_inference.py','r7_math.py','runtime.py','ouro_cached_scoring.py','slcd_r6_scores.py')


def save(path,value):
    p=ROOT/path;assert p.name.startswith('r10_') and p.parent.is_dir() and p.resolve().is_relative_to(ROOT.resolve())
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)


def sources():return {n:sha('scripts/'+n) for n in SOURCES}


def rows(phase):
    protocol=read('configs/r10_PROTOCOL.json');assert sources()==protocol['source_sha256']
    splits={'pilot':['discovery'],'development':['discovery','selection'],
            'confirmation':['confirmation'],'validation':['validation'],'test':['test']}[phase]
    if phase!='pilot':
        label={'development':'DEVELOPMENT','confirmation':'CANDIDATE','validation':'VALIDATION','test':'FINAL_METHOD'}[phase]
        lock=read('configs/r10_'+label+'_LOCK.json')
        assert lock['protocol_sha256']==sha('configs/r10_PROTOCOL.json') and lock['source_sha256']==sources()
        if phase=='test':assert lock['test_access_authorized']
    result=[]
    for split in splits:
        assert sha('data/slcd_'+split+'.jsonl')==protocol['dataset_sha256'][split]
        path='results/r7_'+split+('_development' if split in ['discovery','selection','confirmation'] else '_final')+'.jsonl'
        assert sha(path)==protocol['r7_score_file_sha256'][split]
        receipt=read(path.replace('.jsonl','_COMPLETE.json'));assert receipt['status']=='complete' and receipt['score_file_sha256']==sha(path)
        baseline={r['id']:r for r in jsonlines(path)};data=list(jsonlines('data/slcd_'+split+'.jsonl'))
        assert len(data)==len(baseline)==len({r['id'] for r in data})
        if phase=='pilot':data=data[:64]
        for item in data:
            b=baseline[item['id']]
            assert item['prompt_sha256']==b['prompt_sha256'] and item['gold']==b['gold'] and item['choice_char_lengths']==b['choice_char_lengths']
            result.append(dict(split=split,item=item,baseline_scores=b['baseline_scores']))
    return result


def question_prompt(question,choices):
    assert isinstance(question,str) and len(question)>0 and 2<=len(choices)<=8
    return 'Question: '+question+'\n'+'\n'.join(chr(65+i)+'. '+a for i,a in enumerate(choices))+'\nAnswer:'


def joint_prefix(demonstrations):
    docs=demonstrations['documents'];assert len(docs)==25 and [d['id'] for d in docs]==demonstrations['ids']
    rendered=[]
    for doc in docs:
        i=doc['choices']['label'].index(doc['answerKey'])
        rendered.append(question_prompt(doc['question'],doc['choices']['text'])+' '+chr(65+i))
    return '\n\n'.join(rendered)+'\n\n'


def joint_requests(clean,prefix):
    prompt=prefix+question_prompt(clean['question'],clean['choices'])
    return [(prompt,' '+chr(65+i)) for i in range(len(clean['choices']))]
