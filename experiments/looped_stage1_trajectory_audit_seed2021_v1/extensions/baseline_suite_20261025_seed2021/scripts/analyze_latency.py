"""Question-paired latency bootstrap; repeated timings are not independent N."""
import json
import numpy as np
import pandas as pd
from common import ROOT,CFG,read,save
from analyze_suite import quality,category

def main():
    result=[]
    for model in ['Ouro-1.4B','Ouro-2.6B']:
        rows=read(f'trajectories/actual_halting_{model}.jsonl');assert len(rows)==750
        assert all(r['measured_repeat_quality_matches_offline'] and r['measured_repeat_max_score_error']<.02 for r in rows)
        assert len({r['record_key'] for r in rows})==len(rows)
        parity=json.loads((ROOT/f'results/actual_halting_parity_{model}.json').read_text());assert parity['passed'] and len(parity['cases'])==150
        frame=pd.DataFrame([dict(id=r['id'],method=r['method'],repeat=r['repeat'],seconds=r['elapsed_s'],depth=r['depth'],correct=r['result']['correct'],task=category(r),peak=r['peak_memory_bytes']) for r in rows])
        assert (frame.groupby(['id','method']).size()==5).all()
        median=frame.groupby(['id','method']).agg(seconds=('seconds','median'),depth=('depth','first'),correct=('correct','first'),task=('task','first'),peak=('peak','max')).reset_index()
        for task in ['all','arc_challenge','openbookqa','mmlu']:
            part=median if task=='all' else median[median.task==task]
            baseline=part[part.method=='fixed4'].set_index('id')
            for method in sorted(part.method.unique()):
                aligned=part[part.method==method].set_index('id').loc[baseline.index]
                a=baseline.seconds.to_numpy();b=aligned.seconds.to_numpy();rng=np.random.default_rng(CFG['bootstrap_seed'])
                draws=rng.integers(0,len(a),size=(10000,len(a)));ratios=a[draws].mean(1)/b[draws].mean(1)
                result.append(dict(model=model,task=task,method=method,mean_question_median_s=float(b.mean()),median_question_median_s=float(np.median(b)),p95_question_median_s=float(np.quantile(b,.95)),speedup=float(a.mean()/b.mean()),speedup_ci_low=float(np.quantile(ratios,.025)),speedup_ci_high=float(np.quantile(ratios,.975)),average_depth=float(aligned.depth.mean()),peak_memory_bytes=int(aligned.peak.max()),**quality(baseline.correct,aligned.correct),scope='30 questions, five randomized repetitions; question medians, paired question bootstrap; pretokenized complete MC scoring without cache or service overhead'))
    pd.DataFrame(result).to_csv(ROOT/'results/actual_halting_latency.csv',index=False)
    save('results/actual_halting_analysis.json',dict(passed=True,independent_unit='question',repetitions_per_question=5,bootstrap_resamples=10000,source_parity_all_cases=True,shared_gpu_runtime_caveat='Another user job may run on the same GPU; randomized paired repeats reduce drift but do not establish isolated-serving speedup'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=pd.DataFrame(result);fig,axes=plt.subplots(1,2,figsize=(11,4));labels=['fixed4','fixed2','full_vocab_kl','hidden_relative_l2','native_gate_cdf']
    for ax,model in zip(axes,['Ouro-1.4B','Ouro-2.6B']):
        p=table[(table.model==model)&(table.task=='all')].set_index('method').loc[labels]
        ax.bar(range(len(p)),p.speedup);ax.errorbar(range(len(p)),p.speedup,yerr=np.stack([p.speedup-p.speedup_ci_low,p.speedup_ci_high-p.speedup]),fmt='none',color='black',capsize=3)
        ax.set_xticks(range(len(p)),labels,rotation=30,ha='right');ax.set(title=model,ylabel='Measured MC scoring speedup vs fixed4');ax.axhline(1,color='gray',lw=.7)
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'figures/figure10_actual_halting_latency.{suffix}',dpi=160)
    plt.close(fig)
    print(json.dumps(dict(completed=True,rows=len(result))),flush=True)

if __name__=='__main__':main()
