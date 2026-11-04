"""CPU replay only after complete frozen R10 discovery+selection collection."""
import csv,json,time
import numpy as np
from r7_core import ROOT,read,sha,jsonlines,digest
from r11_scores import all_scores,ELIGIBLE


def main():
    started=time.time();protocol=read('configs/r11_PROTOCOL.json')
    assert {n:sha('scripts/'+n) for n in protocol['source_sha256']}==protocol['source_sha256']
    receipt=read('results/r10_development_initial_COMPLETE.json')
    assert receipt['status']=='complete' and receipt['phase']=='development' and receipt['n']==576
    assert receipt['protocol_sha256']==protocol['r10_protocol_sha256']
    assert receipt['source_sha256']==protocol['r10_source_sha256']
    assert receipt['signature']==digest(dict(protocol_sha256=protocol['r10_protocol_sha256'],source_sha256=protocol['r10_source_sha256'],template='joint_v1',depth=5))
    assert sha(receipt['score_file'])==receipt['score_file_sha256'];aux=list(jsonlines(receipt['score_file']))
    assert len(aux)==len({r['id'] for r in aux})==576
    refs={};items={}
    for split in ['discovery','selection']:
        path='results/r7_'+split+'_development.jsonl'
        assert sha(path)==protocol['r7_score_file_sha256'][split] and sha('data/slcd_'+split+'.jsonl')==protocol['dataset_sha256'][split]
        refs.update({r['id']:r for r in jsonlines(path)});items.update({r['id']:r for r in jsonlines('data/slcd_'+split+'.jsonl')})
    assert set(refs)==set(items)=={r['id'] for r in aux};records=[]
    for row in aux:
        item=items[row['id']];ref=refs[row['id']]
        assert row['gold']==ref['gold']==item['gold'] and row['prompt_sha256']==ref['prompt_sha256']==item['prompt_sha256']
        assert row['signature']==receipt['signature']
        values,diagnostics=all_scores(ref['original_methods'],ref['prior_methods'][0],row)
        base=ref['baseline_scores'];vectors={**base,**{k:v.tolist() for k,v in values.items()}}
        assert np.array_equal(base['M0_R6'],__import__('slcd_r6_scores').selected_score(ref['original_methods'],ref['prior_methods'][0],'r6_prior_projection'))
        preds={k:int(np.argmax(v)) for k,v in vectors.items()};gold=row['gold']
        records.append(dict(id=row['id'],split=row['split'],gold=gold,baseline_scores=base,method_scores={k:v.tolist() for k,v in values.items()},
            diagnostics=diagnostics,predictions=preds,correct={k:v==gold for k,v in preds.items()},protocol_sha256=sha('configs/r11_PROTOCOL.json')))
    output='results/r11_development_scores.jsonl';assert not (ROOT/output).exists()
    with (ROOT/output).open('x',encoding='utf8') as f:
        for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
    table=[]
    for name in records[0]['predictions']:
        b=np.asarray([r['correct']['M0_R6'] for r in records]);v=np.asarray([r['correct'][name] for r in records]);special=np.asarray([r['correct']['M0_R6'] and not r['correct']['fixed'] for r in records])
        table.append(dict(method=name,n=576,correct=int(v.sum()),R6_correct=int(b.sum()),net=int(v.sum()-b.sum()),rescue=int((v&~b).sum()),damage=int((b&~v).sum()),retained_R6_special=int((v&special).sum()),total_R6_special=int(special.sum())))
    with (ROOT/'results/r11_development_results.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    by={r['method']:r for r in table};advance=[n for n in ELIGIBLE if by[n]['correct']>by['M0_R6']['correct'] and by[n]['correct']>by[n+'_prior_norm']['correct'] and by[n]['correct']>by[n+'_shuffle_update']['correct']]
    result=dict(status='complete',n=576,summary=table,eligible_for_confirmation=advance,CPU_only=True,GPU_seconds=0,
        protocol_sha256=sha('configs/r11_PROTOCOL.json'),source_sha256=protocol['source_sha256'],r10_auxiliary_receipt_sha256=sha('results/r10_development_initial_COMPLETE.json'),
        score_file=output,score_file_sha256=sha(output),elapsed_seconds=time.time()-started,
        interpretation='Auxiliary loop difference is a hypothesis, not proven semantic information. Static-native axes are nonselectable attribution controls.')
    (ROOT/'results/r11_DEVELOPMENT_COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
