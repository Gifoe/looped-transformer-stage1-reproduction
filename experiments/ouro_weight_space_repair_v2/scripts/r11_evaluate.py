"""Strict complete-file replay, predeclared gates, paired ARC statistics."""
import csv,json
import numpy as np
from r11_core import ROOT,read,sha,digest,jsonlines,save,gate,rows,extra_sources
from r11_inference import select,joint_baselines


def csvout(path,data):
    with (ROOT/path).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)));w.writeheader();w.writerows(data)


def records(phase):
    protocol,lock,lockpath,names=gate(phase)
    path=f'results/r11_{phase}_selected.jsonl';receipt=read(path.replace('.jsonl','_COMPLETE.json'))
    assert receipt['status']=='complete' and receipt['phase']==phase and receipt['names']==names and receipt['score_file']==path
    assert receipt['protocol_sha256']==sha('configs/r11_PROTOCOL.json') and receipt['definition_lock_sha256']==sha('configs/r11_CANDIDATE_DEFINITION_LOCK.json')
    assert receipt['stage_lock_sha256']==sha(lockpath) and receipt['inference_definition_sha256']==protocol['source_sha256'] and receipt['extra_source_sha256']==extra_sources()
    signature=digest(dict(phase=phase,protocol_sha256=sha('configs/r11_PROTOCOL.json'),lock_sha256=sha(lockpath),names=names,extra_source_sha256=extra_sources()))
    assert receipt['signature']==signature and sha(path)==receipt['score_file_sha256']
    data=list(jsonlines(path));authority={v['item']['id']:v for v in rows(phase)}
    expected={'parity':2,'confirmation':517,'validation':299,'test':1172}[phase]
    assert len(data)==receipt['n']==len(authority)==len({r['id'] for r in data})==expected and set(authority)=={r['id'] for r in data}
    for r in data:
        v=authority[r['id']];item=v['item'];assert r['signature']==signature and r['split']==v['split'] and r['gold']==item['gold'] and r['prompt_sha256']==item['prompt_sha256']
        assert r['baseline_scores']==v['baseline_scores'] and r['actual_auxiliary_depth']==4
        robust='robust_cyclic_loop_axis' in names;assert r['robust_references']==robust
        n=len(item['choices']);assert len(r['cyclic_rotations'])==(n if robust else 1)
        for shift,reference in enumerate(r['cyclic_rotations']):assert reference['order']==[(j+shift)%n for j in range(n)]
        scores,_=select(v['original'],v['prior'],r,names);joint=joint_baselines(r)
        assert set(scores)==set(r['method_scores']) and set(joint)==set(r['joint_baseline_scores'])
        for key,vector in scores.items():assert np.array_equal(vector,r['method_scores'][key]),('Score replay',r['id'],key)
        for key,vector in joint.items():assert np.array_equal(vector,r['joint_baseline_scores'][key])
        vectors={**v['baseline_scores'],**{k:x.tolist() for k,x in scores.items()},**joint}
        assert set(r['predictions'])==set(r['correct'])==set(vectors)
        for key,vector in vectors.items():
            pred=int(np.argmax(vector));assert r['predictions'][key]==pred and r['correct'][key]==(pred==r['gold']),('Outcome replay',r['id'],key)
    return data


def summary(data,name):
    b=np.asarray([r['correct']['M0_R6'] for r in data]);v=np.asarray([r['correct'][name] for r in data]);special=np.asarray([r['correct']['M0_R6'] and not r['correct']['fixed'] for r in data])
    return dict(method=name,n=len(data),correct=int(v.sum()),R6_correct=int(b.sum()),net=int(v.sum()-b.sum()),rescue=int((v&~b).sum()),damage=int((b&~v).sum()),retained_R6_special=int((v&special).sum()),total_R6_special=int(special.sum()))


def table(phase):
    data=records(phase);values=[summary(data,n) for n in data[0]['predictions']]
    csvout('results/r11_'+phase+'_results.csv',values);save('results/r11_'+phase+'_EVALUATED.json',dict(status='passed_exact_replay',n=len(data),phase=phase,summary=values,extra_source_sha256=extra_sources()))
    return data,values


def paired(data,new,reference):
    from scipy.stats import binomtest
    v=np.asarray([r['correct'][new] for r in data]);b=np.asarray([r['correct'][reference] for r in data]);d=v.astype(int)-b.astype(int)
    rescue=int((v&~b).sum());damage=int((~v&b).sum());discord=rescue+damage
    p=1. if discord==0 else float(binomtest(min(rescue,damage),discord,.5,alternative='two-sided').pvalue)
    rng=np.random.default_rng(2021);samples=[]
    for _ in range(200):samples.extend(d[rng.integers(0,len(data),(100,len(data)))].mean(axis=1).tolist())
    low,high=np.quantile(samples,[.025,.975])*100
    return dict(method=new,reference=reference,n=len(data),correct=int(v.sum()),reference_correct=int(b.sum()),accuracy=float(v.mean()),rescue=rescue,damage=damage,
        net=int(d.sum()),delta_pp=float(d.mean()*100),exact_mcnemar_p=p,paired_bootstrap_95_lower_pp=float(low),paired_bootstrap_95_upper_pp=float(high),bootstrap_samples=20000,bootstrap_seed=2021)


def final_report():
    data,t=table('test');lock=read('configs/r11_FINAL_METHOD_LOCK.json');primary=lock['primary'];assert len(data)==1172
    comparisons=[paired(data,primary,k) for k in data[0]['predictions'] if k!=primary]
    csvout('results/r11_arc_paired_statistics.csv',comparisons);csvout('results/r11_arc_full_test_results.csv',t)
    transitions=[]
    for r in data:
        transitions.append(dict(id=r['id'],gold=r['gold'],R6_prediction=r['predictions']['M0_R6'],new_prediction=r['predictions'][primary],
            R6_correct=r['correct']['M0_R6'],new_correct=r['correct'][primary],Fixed_correct=r['correct']['fixed'],
            R6_scores=json.dumps(r['baseline_scores']['M0_R6']),new_scores=json.dumps(r['method_scores'][primary])))
    csvout('results/r11_paired_transitions.csv',transitions)
    m=next(r for r in comparisons if r['reference']=='M0_R6');retention=next(r for r in t if r['method']==primary)
    result=dict(status='complete',primary=primary,n=1172,correct=m['correct'],metrics=m,retention=retention,comparisons=comparisons,
        final_method_lock_sha256=sha('configs/r11_FINAL_METHOD_LOCK.json'),raw_score_file_sha256=sha('results/r11_test_selected.jsonl'),
        history_exposed=True,external_evaluation_authorized=False,semantic_axis_interpretation_established=False)
    save('results/r11_ARC_COMPLETE.json',result)
    text=f"# R11 完整ARC结果\n\n冻结主方法 {primary}：{m['correct']}/1172（{100*m['accuracy']:.4f}%）。R6为{m['reference_correct']}/1172；救回{m['rescue']}、误伤{m['damage']}，净增{m['net']}（{m['delta_pp']:+.4f}pp）。exact McNemar p={m['exact_mcnemar_p']:.6g}，20k配对bootstrap95%CI [{m['paired_bootstrap_95_lower_pp']:.4f},{m['paired_bootstrap_95_upper_pp']:.4f}]pp。\n\n保留R6相对Fixed独有纠错{retention['retained_R6_special']}/{retention['total_R6_special']}。达到751：{m['correct']>=751}。\n\n| 方法 | 正确/1172 | 救回 | 误伤 | 净变化 |\n|---|---:|---:|---:|---:|\n"
    for r in t:text+='| '+' | '.join(str(r[k]) for k in ['method','correct','rescue','damage','net'])+' |\n'
    text+='\n主对比为R11对R6；其他匹配幅度、置乱及静态Native轴均为非晋级对照。按对应控制判断方向或循环差的独立作用，不能只比较弱随机对照。当前辅助轴含有用信息不等于证明其是语义。原R6主提示、完整候选及字符分母保持，但额外联合选项视图改变辅助模板；单独联合评分同列，若更强则不能声称R11具有整体方法优势。历史ARC/dev均已暴露，不是盲测或独立交叉验证，统计区间未校正跨轮研究者适应。\n\n另外两个数据集按用户要求停止。模型权重冻结，无训练/控制器/adapter。部署路径的实际mainK5、N0priorK4及auxiliaryK4时间/峰值见r11_DEPLOYMENT_PROFILE.json；原始主/先验缓存采集和全部新worker成本分开披露，不把缓存重放当部署速度。测试后未改赢家。\n'
    (ROOT/'r11_ARC_REPORT.md').write_text(text,encoding='utf8');return result
