"""Own finite-search contract; no external benchmark execution is authorized."""
import json
from r7_core import ROOT,read,sha,digest,jsonlines

SOURCES=('r9_core.py','r9_scores.py','r9_worker.py','r8_scores.py',
         'r7_core.py','r7_inference.py','runtime.py','ouro_cached_scoring.py','slcd_r6_scores.py')


def save(path,value):
    p=ROOT/path
    assert p.name.startswith('r9_') and p.parent.is_dir() and p.resolve().is_relative_to(ROOT.resolve())
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)


def sources():return {n:sha('scripts/'+n) for n in SOURCES}


def rows(phase):
    protocol=read('configs/r9_PROTOCOL.json')
    assert sources()==protocol['source_sha256']
    splits={'pilot':['discovery'],'development':['discovery','selection'],
            'confirmation':['confirmation'],'validation':['validation'],'test':['test']}[phase]
    if phase in ['confirmation','validation','test']:
        lock=read('configs/r9_'+{'confirmation':'CANDIDATE','validation':'VALIDATION','test':'FINAL_METHOD'}[phase]+'_LOCK.json')
        assert lock['protocol_sha256']==sha('configs/r9_PROTOCOL.json')
        assert lock['source_sha256']==sources()
        if phase=='test':assert lock['test_access_authorized']
    result=[]
    for split in splits:
        assert sha('data/slcd_'+split+'.jsonl')==protocol['dataset_sha256'][split]
        path='results/r7_'+split+('_development' if split in ['discovery','selection','confirmation'] else '_final')+'.jsonl'
        assert sha(path)==protocol['r7_score_file_sha256'][split]
        receipt=read(path.replace('.jsonl','_COMPLETE.json'))
        assert receipt['status']=='complete' and receipt['score_file_sha256']==sha(path)
        baseline={r['id']:r for r in jsonlines(path)}
        data=list(jsonlines('data/slcd_'+split+'.jsonl'))
        assert len(data)==len(baseline)==len({r['id'] for r in data})
        if phase=='pilot':data=data[:64]
        for item in data:
            b=baseline[item['id']]
            assert item['prompt_sha256']==b['prompt_sha256'] and item['gold']==b['gold']
            assert item['choice_char_lengths']==b['choice_char_lengths']
            result.append(dict(split=split,item=item,baseline_scores=b['baseline_scores']))
    return result


def reverse_prefix(demonstrations):
    docs=demonstrations['documents'];assert len(docs)==25
    assert [d['id'] for d in docs]==demonstrations['ids']
    rendered=[]
    for doc in docs:
        index=doc['choices']['label'].index(doc['answerKey'])
        rendered.append('Answer: '+doc['choices']['text'][index]+'\nQuestion: '+doc['question'])
    return '\n\n'.join(rendered)+'\n\n'


def reverse_requests(clean,prefix):
    """Accesses no gold or original source_doc/answerKey."""
    q=clean['question'];choices=clean['choices']
    assert isinstance(q,str) and len(q)>0 and len(choices)>=2
    return [(prefix+'Answer: '+answer+'\nQuestion:',' '+q) for answer in choices]
