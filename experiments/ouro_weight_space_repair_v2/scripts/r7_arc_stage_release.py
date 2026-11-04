"""Complete ARC snapshot and lossless raw-score release during external runs.

This does not declare the whole conditional experiment complete. No inference,
selection or model mutation takes place here.
"""
import hashlib,lzma,time
from datetime import datetime,timezone
from r7_core import ROOT,read,save,sha,csvout


def main():
    from r7_evaluate import rows
    arc=read('results/r7_ARC_COMPLETE.json');accept=read('results/r7_TEST_ACCEPTANCE.json')
    assert arc['status']=='complete' and arc['n']==1172 and accept['status']=='passed'
    assert arc['score_file_sha256']==accept['new_score_file_sha256']==sha('results/r7_test_final.jsonl')
    assert arc['final_method_lock_sha256']==accept['final_method_lock_sha256']==sha('configs/r7_FINAL_METHOD_LOCK.json')
    data=rows('test');primary=arc['primary'];m=arc['metrics'];strata=[]
    for f,b,label in [(True,True,'Fixed_and_R6_correct'),(False,True,'R6_only_correct'),(True,False,'Fixed_only_correct'),(False,False,'Fixed_and_R6_wrong')]:
        group=[r for r in data if r['correct']['fixed']==f and r['correct']['M0_R6']==b];n=sum(r['correct'][primary] for r in group)
        strata.append(dict(stratum=label,n=len(group),new_correct=n,new_wrong=len(group)-n,net_vs_R6=n-(len(group) if b else 0)))
    csvout('results/r7_rescue_retention.csv',strata)
    by_stratum={r['stratum']:r for r in strata}
    profile=read('results/r7_DEPLOYMENT_PROFILE.json');assert profile['status']=='complete'
    cancelled=(ROOT/'logs/r7_SCOPE_CHANGE_EXTERNAL_STOP.json').exists()
    text=f'''# R7 ARC阶段交付：{'外部评估已按用户要求停止' if cancelled else '外部评估尚未结束'}

UTC快照：{datetime.now(timezone.utc).isoformat()}。冻结主方法：{primary}；全部工作仍在原ouro_weight_space_repair_v2文件夹。

完整ARC-Challenge：**{m['correct']}/1172（{100*m['accuracy']:.4f}%）**；R6为736/1172，净增{m['net']}题（{m['accuracy_delta_pp']:+.4f}pp）。纠错{m['rescue']}、误伤{m['damage']}；exact McNemar p={m['exact_mcnemar_p']:.6g}；题级20,000次配对bootstrap区间[{m['paired_bootstrap_95_lower_pp']:.4f}, {m['paired_bootstrap_95_upper_pp']:.4f}]pp。

**未达到751，也未得到显著增益证据。** R6相对Fixed的{m['total_R6_vs_Fixed_rescues']}道独有纠错保留{m['retained_R6_vs_Fixed_rescues']}道。{m['rescue']}道救回中{by_stratum['Fixed_only_correct']['new_correct']}道是恢复Fixed本已答对的题，{by_stratum['Fixed_and_R6_wrong']['new_correct']}道是Fixed/R6均错的题；{m['damage']}道误伤包括{by_stratum['R6_only_correct']['new_wrong']}道R6独有纠错和{by_stratum['Fixed_and_R6_correct']['new_wrong']}道两者本来都正确的题。当前证据不足以把本方法视为更可靠的选择性升级。

| 原Fixed/R6状态 | 题数 | 新正确 | 新错误 | 相对R6净变化 |
| --- | --- | --- | --- | --- |
'''
    text+='\n'.join('| '+' | '.join(str(r[k]) for k in ['stratum','n','new_correct','new_wrong','net_vs_R6'])+' |' for r in strata)
    text+=f'''

完整1172题的冻结R6逐候选分数和N0先验总似然均独立核验为零误差。最终方法锁SHA256：`{arc['final_method_lock_sha256']}`。原始评分记录SHA256：`{arc['score_file_sha256']}`。测试后没有另选主方法。

实际K5/K4部署样本12题，后11题均值{profile['warm_mean_question_seconds']:.4f}秒；同题R6为{profile['R6_warm_mean_question_seconds']:.4f}秒，比值{profile['warm_time_ratio_vs_R6']:.4f}。这是一组顺序测量的小样本，不能据毫秒差异声称硬件速度优势。原始无KV、无mask行为保持；零校正完整续写恢复Fixed/A1的真实模型检查已通过。
'''
    oq=ROOT/'results/r7_openbookqa_COMPLETE.json'
    if oq.exists():
        o=read('results/r7_openbookqa_COMPLETE.json');assert o['status']=='complete' and o['n']==500 and o['primary']==primary
        om=o['metrics'];text+=f"\n完整OpenBookQA：{o['correct']}/500，R6为{om['reference_correct']}/500，净变化{om['net']}题，纠错{om['rescue']}、误伤{om['damage']}，p={om['exact_mcnemar_p']:.6g}。\n"
    if cancelled:
        text+='\n用户随后要求继续寻找方法、暂停另外两个数据集。OpenBookQA在该指令前已完成，保留其完整负结果；HellaSwag的工作进程已停止，部分记录只作运行记录，不报告完整准确率，也不用于后续方法选择。后续只在ARC开发数据上探索，R7公式及本次740/1172结果保持冻结。\n'
    else:
        text+='\nARC超过736，按预设门槛继续同一冻结公式的完整HellaSwag10042及OpenBookQA500；尚未全部完成的外部结果不能当作本轮最终结论。\n'
    text+='\n原始test逐题评分压缩分片及SHA见artifacts/r7_ARC_STAGE_INDEX.json；训练、验证及转移表均已保留。数据历史已暴露，本次不是新盲测。\n'
    (ROOT/'r7_ARC_STAGE_REPORT.md').write_text(text,encoding='utf8')
    raw=(ROOT/'results/r7_test_final.jsonl').read_bytes();packed=lzma.compress(raw,preset=6)
    assert lzma.decompress(packed)==raw
    archive='artifacts/r7_arc_test_scores.jsonl.xz';(ROOT/archive).write_bytes(packed);parts=[]
    for start in range(0,len(packed),4*1024*1024):
        part=packed[start:start+4*1024*1024];name=f'artifacts/r7_arc_scores.part{len(parts):02d}.xz';(ROOT/name).write_bytes(part)
        parts.append(dict(path=name,bytes=len(part),sha256=hashlib.sha256(part).hexdigest()))
    save('artifacts/r7_ARC_STAGE_INDEX.json',dict(status='verified_complete_ARC_snapshot',complete_experiment=False,external_cancelled_by_user=cancelled,created_unix=time.time(),primary=primary,
        release_source_sha256=sha('scripts/r7_arc_stage_release.py'),
        correct=arc['correct'],n=1172,final_method_lock_sha256=arc['final_method_lock_sha256'],
        raw=dict(path='results/r7_test_final.jsonl',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),
        compressed=dict(path=archive,bytes=len(packed),sha256=hashlib.sha256(packed).hexdigest()),parts=parts,
        restoration='Concatenate listed part bytes in order, validate compressed SHA256, then lzma.decompress; validate raw SHA256 before using as results/r7_test_final.jsonl',
        contents='All1172 raw numerical score rows with per-token scalar diagnostics; no corpus-wide vocabulary logits or raw hidden tensors'))
    print('R7_ARC_STAGE_RELEASE_VERIFIED',len(raw),len(packed),len(parts),flush=True)


if __name__=='__main__':main()
