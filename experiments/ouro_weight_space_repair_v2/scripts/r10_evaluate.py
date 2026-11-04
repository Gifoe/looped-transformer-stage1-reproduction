"""CPU attribution/gates and paired statistics. No alternative held-out winners."""
import argparse,csv,json,math
import numpy as np
from r10_core import ROOT,read,sha,jsonlines,save
from r10_auxiliary import baselines
from r9_scores import evidence
from r9_worker import summary


def control(name):
    return ('joint4_raw_shuffled_control' if name=='joint4_fusion' else
            ('balanced4' if name=='balanced4_protected' else 'jointFixed' if name=='jointFixed_protected' else 'joint4')+'_shuffled_control')


def csvout(path,data):
    with (ROOT/path).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for v in data for k in v)));w.writeheader();w.writerows(data)


def records(phase):
    path='results/r10_'+phase+('_initial' if phase in ['pilot','development'] else '_selected')+'.jsonl'
    r=read(path.replace('.jsonl','_COMPLETE.json'));assert r['status']=='complete' and sha(path)==r['score_file_sha256']
    data=list(jsonlines(path));assert len(data)==r['n']==len({v['id'] for v in data})
    for row in data:
        for k,v in baselines(row).items():
            row['predictions'][k]=int(v.argmax());row['correct'][k]=int(v.argmax())==row['gold']
        if phase in ['pilot','development']:
            r6=np.asarray(row['baseline_scores']['M0_R6']);u,_=evidence(r6,row['auxiliary_sum_scores']['native_k4']);p=np.random.default_rng(2021).permutation(len(u))
            n='joint4_raw_shuffled_control';row['predictions'][n]=int((r6+u[p]).argmax());row['correct'][n]=row['predictions'][n]==row['gold']
    return data


def table(phase):
    data=records(phase);result=[summary(data,n) for n in data[0]['predictions']]
    csvout('results/r10_'+phase+'_summary.csv',result)
    save('results/r10_'+phase+'_EVALUATED.json',dict(status='complete',phase=phase,n=len(data),summary=result,
        evaluator_sha256=sha('scripts/r10_evaluate.py'),post_pilot_audit_sha256=sha('configs/r10_POST_PILOT_AUDIT.json')))
    return data,result


def paired(data,new,old):
    x=np.asarray([r['correct'][new] for r in data]);y=np.asarray([r['correct'][old] for r in data]);delta=x.astype(int)-y.astype(int)
    rescue=int((x&~y).sum());damage=int((~x&y).sum());n=rescue+damage
    from scipy.stats import binomtest
    p=1. if n==0 else float(binomtest(min(rescue,damage),n,.5,alternative='two-sided').pvalue)
    rng=np.random.default_rng(2021);boots=[]
    for _ in range(200):boots.extend(delta[rng.integers(0,len(data),size=(100,len(data)))].mean(axis=1).tolist())
    lo,hi=np.quantile(boots,[.025,.975])*100
    return dict(method=new,reference=old,n=len(data),correct=int(x.sum()),reference_correct=int(y.sum()),accuracy=float(x.mean()),
        rescue=rescue,damage=damage,net=int(delta.sum()),delta_pp=float(delta.mean()*100),exact_mcnemar_p=p,
        paired_bootstrap_95_lower_pp=float(lo),paired_bootstrap_95_upper_pp=float(hi),bootstrap_samples=20000,bootstrap_seed=2021)


def final_report():
    data,t=table('test');lock=read('configs/r10_FINAL_METHOD_LOCK.json');primary=lock['primary']
    assert lock['test_access_authorized'] and len(data)==1172
    comparisons=[paired(data,primary,k) for k in data[0]['predictions'] if k!=primary and ('control' not in k and not k.startswith('diagnostic_'))]
    csvout('results/r10_arc_paired_statistics.csv',comparisons)
    m=next(r for r in comparisons if r['reference']=='M0_R6');joint=next(r for r in comparisons if r['reference']=='joint4_only')
    stats=dict(status='complete',primary=primary,n=1172,correct=m['correct'],metrics=m,comparisons=comparisons,
        final_method_lock_sha256=sha('configs/r10_FINAL_METHOD_LOCK.json'),raw_score_file_sha256=sha('results/r10_test_selected.jsonl'),
        history_exposed=True,external_evaluation_authorized=False,joint_format_is_auxiliary_protocol_change=True,
        causal_loop_mechanism_established=False)
    save('results/r10_ARC_COMPLETE.json',stats)
    lines=['# R10 ARC完整结果与归因限制','',f"冻结主方法：{primary}。完整ARC：{m['correct']}/1172 ({100*m['accuracy']:.4f}%)；R6为{m['reference_correct']}/1172，净增{m['net']}，救回{m['rescue']}、误伤{m['damage']}。McNemar p={m['exact_mcnemar_p']:.6g}，20k配对bootstrap95%CI [{m['paired_bootstrap_95_lower_pp']:.4f},{m['paired_bootstrap_95_upper_pp']:.4f}]pp。",'',
        '| 方法 | 正确/1172 |','|---|---:|']
    for row in t:
        if 'control' not in row['method'] and not row['method'].startswith('diagnostic_'):lines.append(f"| {row['method']} | {row['correct']} |")
    lines+=['',f"与只用联合Native评分相比，本冻结融合方法净变化{joint['net']}题。提示视图发生改变：原R6主路径仍是原固定25-shot完整答案续写与acc_norm，辅助视图将同25示例和当前题渲染为含全部选项的字母答题。不能把格式增益归因于新的循环机制，也不能声称与原单流协议等价或等计算成本。",'',
        '联合选项和位置平衡都是已有方法；本轮检验其与Ouro/R6纠错的互补性。联合-only基线不参与主方法选择；在64题pilot观察后、扩大开发数据前登记为归因对照。所有ARC数据已在历史研究中暴露，本次不是盲测或独立交叉验证。',
        '', '另两个数据集按用户要求保持停止。R6/R7历史代码、锁、结果与预算保持；本轮算力继承同一个6小时上限。全代码与原始数值留在同一个实验文件夹。']
    (ROOT/'r10_ARC_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return stats


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','development','confirmation','validation','test']);a=p.parse_args()
    print(json.dumps(final_report() if a.phase=='test' else table(a.phase)[1]),flush=True)
