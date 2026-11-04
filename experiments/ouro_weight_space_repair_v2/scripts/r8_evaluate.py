"""Finite CPU method comparisons. All inference arguments are label-free vectors."""
import argparse,time
import numpy as np
from r8_core import ROOT,sha,read,save,csvout,load_rows
from r8_scores import selected_score


def summary(data,method):
    base=np.asarray([r['base_correct'] for r in data]);new=np.asarray([r['correct'][method] for r in data])
    special=np.asarray([r['base_correct'] and not r['fixed_correct'] for r in data])
    return dict(method=method,n=len(data),correct=int(new.sum()),R6_correct=int(base.sum()),rescue=int((new&~base).sum()),damage=int((base&~new).sum()),
        net=int(new.sum()-base.sum()),retained_R6_vs_Fixed_rescues=int((new&special).sum()),total_R6_vs_Fixed_rescues=int(special.sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','confirmation','validation','test'],required=True);p.add_argument('--job',default='');a=p.parse_args();start=time.time()
    assert not a.job or a.job.replace('_','').isalnum()
    cfg=read('configs/r8_METHODS.json');names=list(cfg['methods'])
    if a.split=='validation':names=read('configs/r8_VALIDATION_LOCK.json')['candidates']
    elif a.split=='test':names=[read('configs/r8_FINAL_METHOD_LOCK.json')['primary']]
    if a.split in ['confirmation','validation','test']:
        lock=read('configs/r8_FINAL_METHOD_LOCK.json' if a.split=='test' else 'configs/r8_CANDIDATE_LOCK.json')
        assert lock['methods_sha256']==sha('configs/r8_METHODS.json')
        for n,value in lock['source_sha256'].items():assert sha('scripts/'+n)==value
    splits=['discovery','selection'] if a.split=='development' else [a.split];records=[]
    need_p5=any(n.startswith('late_') or n=='matched_late' for n in names)
    for split in splits:
        for row in load_rows(split,need_p5=need_p5):
            baseline=row['baseline_scores'];preds={k:int(np.argmax(v)) for k,v in baseline.items()};scores={};diagnostics={}
            if row['historical_M1'] is not None:preds['historical_R7_M1']=int(np.argmax(row['historical_M1']))
            preds['historical_R7_M2']=int(np.argmax(row['m2']))
            for name in names:
                values,d=selected_score(row['a'],row['r'],row['m2'],row['late'],row['prior_late'],name,fixed=row['baseline_scores']['fixed'])
                scores[name]=values.tolist();preds[name]=int(values.argmax());diagnostics[name]=d
            gold=row['gold'];records.append(dict(id=row['id'],split=split,gold=gold,predictions=preds,
                correct={k:v==gold for k,v in preds.items()},method_scores=scores,diagnostics=diagnostics,
                base_correct=preds['M0_R6']==gold,fixed_correct=preds['fixed']==gold,source_score_sha256=row['source_score_sha256']))
    stem=f'results/r8_{a.split}'+('_'+a.job if a.job else '');output=stem+'_scores.jsonl'
    assert not (ROOT/output).exists(),'An existing attempt must be inspected, not overwritten'
    with (ROOT/output).open('x',encoding='utf8') as stream:
        import json
        for row in records:stream.write(json.dumps(row,allow_nan=False)+'\n')
    table=[]
    groups=[(s,[r for r in records if r['split']==s]) for s in splits]
    if len(splits)>1:groups.append((a.split,records))
    for label,subset in groups:
        if not subset:continue
        for name in list(subset[0]['predictions']):table.append(dict(split=label,**summary(subset,name)))
    csvout(stem+'_results.csv',table)
    save(stem+'_COMPLETE.json',dict(status='complete',n=len(records),methods=names,score_file=output,score_file_sha256=sha(output),
        methods_sha256=sha('configs/r8_METHODS.json'),source_sha256={n:sha('scripts/'+n) for n in ['r8_scores.py','r8_core.py','r8_evaluate.py']},
        elapsed_seconds=time.time()-start,CPU_only=True,summary=[r for r in table if r['split']==a.split]))
    print('R8_COMPLETE',a.split,[r for r in table if r['split']==a.split],flush=True)


if __name__=='__main__':main()
