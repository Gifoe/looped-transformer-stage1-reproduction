import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from analyze import ROOT,paired_ci

def main():
    data={}
    for method in ['A0','A1','A2','A3','A4']:
        rows=[json.loads(l) for l in (ROOT/'results'/('qwen_'+method+'_test.jsonl')).read_text(encoding='utf-8').splitlines()]
        assert len(rows)==600
        data[method]={x['id']:x for x in rows};assert len(data[method])==len(rows)
    ids=list(data['A0']);assert all(set(data[k])==set(ids) for k in data)
    groups={'all':ids}
    for uid in ids:
        task=data['A0'][uid]['task'];groups.setdefault(task,[]).append(uid)
        if task.startswith('mmlu_'):groups.setdefault('mmlu',[]).append(uid)
    table=[];examples=[];pairs=[]
    for group,subset in groups.items():
        for first,second in combinations(data,2):
            a=np.array([data[first][uid]['correct'] for uid in subset]);b=np.array([data[second][uid]['correct'] for uid in subset])
            pairs.append(dict(task=group,reference=first,method=second,n=len(subset),wrong_to_correct=int((~a&b).sum()),correct_to_wrong=int((a&~b).sum()),**paired_ci(a,b)))
        base=np.array([data['A0'][uid]['correct'] for uid in subset])
        for method,aligned in data.items():
            rows=[aligned[uid] for uid in subset];correct=np.array([x['correct'] for x in rows]);cfg=rows[0]['configuration']
            choice_kl=[float(np.sum(np.array(aligned[uid]['choice_probabilities'])*(np.log(aligned[uid]['choice_probabilities'])-np.log(data['A0'][uid]['choice_probabilities'])))) for uid in subset]
            transition=dict(correct_to_wrong=int((base&~correct).sum()),wrong_to_correct=int((~base&correct).sum()),correct_to_correct=int((base&correct).sum()),wrong_to_wrong=int((~base&~correct).sum()))
            table.append(dict(task=group,method=method,n=len(rows),accuracy=float(correct.mean()),**transition,**paired_ci(base,correct),
                              mean_gold_nll_per_token=float(np.mean([x['gold_nll_per_token'] for x in rows])),
                              mean_full_vocab_entropy=float(np.mean([x['full_vocab_stats']['entropy'] for x in rows])),
                              mean_gold_choice_logp=float(np.mean([x['gold_choice_logp'] for x in rows])),
                              mean_choice_kl_vs_original=float(np.mean(choice_kl)),
                              total_elapsed_s=sum(x['elapsed_s'] for x in rows),mean_elapsed_s=float(np.mean([x['elapsed_s'] for x in rows])),
                              peak_memory_bytes=max(x['peak_memory_bytes'] for x in rows),
                              theoretical_layer_compute_ratio=1+max(0,cfg['K']-1)*4/36))
            if group=='all' and method!='A0':
                for direction,mask in [('wrong_to_correct',~base&correct),('correct_to_wrong',base&~correct)]:
                    for index in np.flatnonzero(mask)[:5]:examples.append(dict(direction=direction,method=method,baseline=data['A0'][subset[index]],variant=rows[index]))
    pd.DataFrame(table).to_csv(ROOT/'results/qwen_metrics.csv',index=False)
    pd.DataFrame(pairs).to_csv(ROOT/'results/qwen_pairwise_metrics.csv',index=False)
    (ROOT/'results/qwen_failure_cases.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in examples),encoding='utf-8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=pd.DataFrame(table);fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for task in ['arc_challenge','openbookqa','mmlu']:
        rows=frame[frame.task==task]
        labels={'arc_challenge':'ARC-Challenge','openbookqa':'OpenBookQA','mmlu':'MMLU math/logic'}
        axes[0].plot(rows.method,rows.accuracy,'o-',label=labels[task])
        axes[1].plot(rows.method,rows.mean_gold_nll_per_token,'o-',label=labels[task])
    for ax in axes:ax.set_xticks(range(5),['A0\nOriginal','A1\nNaive K2','A2\nNaive K6','A3\nDamped K6','A4\nAnchored K6'],fontsize=8)
    axes[0].set(ylabel='Accuracy',xlabel='Fixed loop strategy');axes[0].legend()
    axes[1].set(ylabel='Gold NLL per continuation token',xlabel='Fixed loop strategy')
    fig.suptitle('Independent published-recipe reproduction, zero-shot')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/('figure1_loop_strategy.'+ext),dpi=180)
    plt.close(fig)
    print(frame[frame.task=='all'].to_json(orient='records'))

if __name__=='__main__':main()
