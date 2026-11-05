"""Independent completed-run replay; refuses partial effects and never uses GPU."""
from pathlib import Path
import collections
import csv
import hashlib
import itertools
import json

import numpy as np

P=Path(__file__).resolve().parent
ROOT=P.parent
H=lambda b:hashlib.sha256(b).hexdigest()
read=lambda n:json.loads((P/n).read_text(encoding='utf8'))
lock=read('EXECUTION_LOCK.json')
sup=read('SUPERVISOR.json')
complete=read('COMPLETE.json')
budget=read('gpu_budget.json')
validation=read('numerical_validation.json')
analysis=read('ANALYSIS.json')
assert sup['status']=='complete' and sup['exit_code']==0 and not sup['overrun'] and not sup['timed_out']
assert complete['status']=='complete_all218_tasks' and complete['tasks']==218 and complete['native_heads']==872 and complete['head_prediction_positions']==1448
assert analysis['status']=='complete24_all_conditions_numeric_before_labels' and validation['status']=='passed_all_numeric_before_labels' and validation['full_intervention_matrix']
assert complete['protocol_sha256']==sup['protocol_sha256']==H((P/'EXECUTION_LOCK.json').read_bytes())
oldtotal=0.
for path,digest in lock['inherited_ledgers'].items():
    raw=(ROOT/path).read_bytes();assert H(raw)==digest
    oldtotal+=json.loads(raw)['used_seconds']
assert len(lock['inherited_ledgers'])==20 and oldtotal==20889.326696651056
assert 0<sup['gpu_worker_seconds']<=680 and budget['used_seconds']==sup['gpu_worker_seconds']
assert oldtotal+budget['used_seconds']==budget['total_used_seconds']<=21600
assert 21600-budget['total_used_seconds']==budget['remaining_seconds']
assert complete['parameter_hashes_before']==complete['parameter_hashes_after'] and len(complete['parameter_hashes_before'])==269
assert complete['parameter_hashes_after']==json.loads((ROOT/'recurrent_decision_audit_v1/a4_COMPLETE.json').read_text(encoding='utf8'))['parameter_hashes_after']
for filename,key in [('SCORES.jsonl','SCORES_sha256'),('VECTORS.npz','VECTORS_sha256'),('VECTOR_INDEX.json','VECTOR_INDEX_sha256')]:
    assert H((P/filename).read_bytes())==complete[key]
for filename,digest in lock['source_sha256'].items():assert H((P/filename).read_bytes())==digest
for path,digest in lock['input_sha256'].items():
    if (ROOT/path).exists():assert H((ROOT/path).read_bytes())==digest

plan=read('INFERENCE_TASK_PLAN.json')
assert H((P/'INFERENCE_TASK_PLAN.json').read_bytes())==complete['plan_sha256']==lock['plan_sha256']
raw=[json.loads(line) for line in (P/'SCORES.jsonl').read_text(encoding='utf8').splitlines()]
assert len(raw)==218 and [x['task_key'] for x in raw]==[x['task_key'] for x in plan['tasks']]
index=read('VECTOR_INDEX.json')
npz=np.load(P/'VECTORS.npz',allow_pickle=False)
assert set(npz.files)==set(index)
for key,meta in index.items():
    array=npz[key]
    assert array.dtype==np.uint16 and list(array.shape)==meta['shape']
    assert H((meta['kind']+str(array.shape)).encode()+array.tobytes())==key
    assert np.isfinite((array.astype(np.uint32)<<16).view(np.float32)).all()
bycid={}
max_norm_error=max_lp_reference_error=0.
for record,task in zip(raw,plan['tasks']):
    assert set(record['scores'])=={c['consumer_id'] for c in task['consumers']}
    length=len(task['positions'])
    for k in range(4):
        assert npz[record['state_keys'][k]].shape==(1,length,2048)
        head=(npz[record['head_keys'][k]].astype(np.uint32)<<16).view(np.float32)
        assert head.shape==(length,49152)
        z=head.astype(np.float64);maximum=z.max(axis=1)
        norm=maximum+np.log(np.exp(z-maximum[:,None]).sum(axis=1))
        max_norm_error=max(max_norm_error,float(np.max(np.abs(norm-np.asarray(record['fullV_log_normalizer_FP64'][k])))))
        for c in task['consumers']:
            value=record['scores'][c['consumer_id']][k]
            target=c['target_ids'];lp=np.asarray(value['token_logprobs'])
            actual=head[np.arange(length),target]
            assert np.array_equal(actual,value['raw_target_logits'])
            assert np.array_equal(lp,lp.astype(np.float32).astype(float))
            assert value['char_denominator']==c['char_denominator'] and value['char_normalized']==value['sum_logprob']/c['char_denominator']
            max_lp_reference_error=max(max_lp_reference_error,float(np.max(np.abs(lp-(actual-norm)))))
    for cid,scores in record['scores'].items():assert cid not in bycid;bycid[cid]=scores
assert len(bycid)==784 and max_norm_error<=1e-10 and max_lp_reference_error<=5e-5
for cid,reference in plan['historical_endpoint_references'].items():
    assert bycid[cid][0]['sum_logprob']==reference['K1'] and bycid[cid][3]['sum_logprob']==reference['K4']
assert len(complete['historical_parity_checks'])==192

# Only a complete, source/parameter/raw-byte-validated run reaches labels.
side=read('CONSTRUCTOR_WITH_LABELS.json')
ids=[q['id'] for q in side['cohort']]
assert ids==read('ROOT_SHA_COHORT.json')['pilot24_ids']
group={(g['id'],g['condition']):g for g in side['groups']}
assert len(group)==196

def vector(g,k,field='sum_logprob'):
    return np.asarray([bycid[c['consumer_id']][k-1][field] for c in g['consumers']])

def pred(g,k,field='sum_logprob'):
    displayed=int(vector(g,k,field).argmax())
    return displayed,g['order_to_original'][displayed]

with (P/'native_predictions.csv').open(encoding='utf8',newline='') as f:native=list(csv.DictReader(f))
assert len(native)==784
for row in native:
    g=group[row['id'],row['condition']];k=int(row['K']);v=vector(g,k)
    assert np.array_equal(v,json.loads(row['raw_sum_scores']))
    assert np.array_equal(vector(g,k,'char_normalized'),json.loads(row['char_normalized_scores']))
    displayed,content=pred(g,k)
    assert int(row['display_prediction'])==displayed and int(row['original_content_prediction'])==content
    assert (row['target_match']=='True')==(content==g['expected_original_content_index'])
    j=g['expected_display_index'];assert float(row['raw_gold_margin'])==float(v[j]-np.delete(v,j).max())
with (P/'permutation_transitions.csv').open(encoding='utf8',newline='') as f:permutation=list(csv.DictReader(f))
assert len(permutation)==288
for row in permutation:
    q=row['id'];k=int(row['K']);g=group[q,row['condition']]
    bp,bc=pred(group[q,'P0'],k);p,c=pred(g,k)
    assert int(row['new_display_prediction'])==p and int(row['new_content_prediction'])==c
    assert (row['content_equivariant']=='True')==(bc==c)
    assert (row['display_letter_invariant']=='True')==(bp==p)
    assert (row['P0_predicted_content_moved']=='True')==(g['order_to_original'].index(bc)!=bp)

def I(q,variant,family,k):
    a=group[q,'P0' if family=='Q' else 'N_original']
    b=group[q,('Q_' if family=='Q' else 'N_')+variant]
    return vector(a,k)-vector(b,k)

with (P/'distractor_effects.csv').open(encoding='utf8',newline='') as f:effects=list(csv.DictReader(f))
assert len(effects)==672  # Q and N are paired columns in each option/pair row.
for row in effects:
    q=row['id'];variant=row['variant'];k=int(row['K']);i=int(row['option_i'])
    iq=I(q,variant,'Q',k);inn=I(q,variant,'N',k)
    if row['record_type']=='unchanged_option':
        for name,value in [('I_Q',iq[i]),('I_N',inn[i]),('I_QminusN',iq[i]-inn[i])]:assert float(row[name])==float(value)
        if k<4:assert float(row['delta_to_next_I_Q'])==float(I(q,variant,'Q',k+1)[i]-iq[i])
    else:
        j=int(row['option_l']);jq=iq[i]-iq[j];jn=inn[i]-inn[j]
        for name,value in [('J_Q',jq),('J_N',jn),('J_QminusN',jq-jn)]:assert float(row[name])==float(value)
        if k<4:
            qnext=I(q,variant,'Q',k+1);nnext=I(q,variant,'N',k+1)
            assert float(row['delta_to_next_J_QminusN'])==float((qnext[i]-qnext[j])-(nnext[i]-nnext[j])-(jq-jn))

with (P/'question_level_statistics.csv').open(encoding='utf8',newline='') as f:statistics=list(csv.DictReader(f))
statmap={(r['metric'],r['condition'],r['K']):r for r in statistics}
schedule={}
checked_stats=0
approximate_stats=0
max_stat_error=0.

def checkstat(key,values,qids,approximate=False):
    global checked_stats,approximate_stats,max_stat_error
    r=statmap[key];v=np.asarray(values,float);n=len(v)
    assert int(r['n'])==n and json.loads(r['ids'])==qids
    if n:
        if n not in schedule:schedule[n]=np.random.default_rng(2021).integers(0,n,(20000,n))
        ci=np.quantile(v[schedule[n]].mean(axis=1),[.025,.975])
        actual=np.asarray([float(r['mean']),float(r['ci95_low']),float(r['ci95_high'])])
        expected=np.asarray([float(v.mean()),float(ci[0]),float(ci[1])])
        error=float(np.max(np.abs(actual-expected)));max_stat_error=max(max_stat_error,error)
        assert error<=(1e-12 if approximate else 0.),(key,error)
    if approximate:approximate_stats+=1
    else:checked_stats+=1

counts={}
for condition in sorted({g['condition'] for g in side['groups']}):
    qids=[q for q in ids if (q,condition) in group]
    for field,label in [('sum_logprob','raw'),('char_normalized','char_norm')]:
        correct={k:[int(pred(group[q,condition],k,field)[1]==group[q,condition]['expected_original_content_index']) for q in qids] for k in range(1,5)}
        if field=='sum_logprob':counts[condition]=[sum(correct[k]) for k in range(1,5)]
        for k in range(1,5):checkstat(('target_match_'+label,condition,str(k)),correct[k],qids)
        for k in range(1,4):checkstat(('adjacent_target_match_delta_'+label,condition,str(k)+'to'+str(k+1)),[a-b for a,b in zip(correct[k+1],correct[k])],qids)
        checkstat(('K4_minus_K1_target_match_'+label,condition,'1to4'),[a-b for a,b in zip(correct[4],correct[1])],qids)
for condition in ('P1','P2','P3'):
    for metric in ('content_equivariant','display_letter_invariant'):
        values={}
        for k in range(1,5):
            rr=[r for r in permutation if r['condition']==condition and int(r['K'])==k]
            values[k]=[int(r[metric]=='True') for r in rr]
            checkstat((metric,condition,str(k)),values[k],ids)
            moved=[r for r in rr if r['P0_predicted_content_moved']=='True']
            checkstat((metric+'_moved_P0_content_subset',condition,str(k)),[int(r[metric]=='True') for r in moved],[r['id'] for r in moved])
        for k in range(1,4):checkstat(('adjacent_'+metric+'_delta',condition,str(k)+'to'+str(k+1)),[a-b for a,b in zip(values[k+1],values[k])],ids)
        checkstat(('K4_minus_K1_'+metric,condition,'1to4'),[a-b for a,b in zip(values[4],values[1])],ids)

with (P/'content_letter_readout.csv').open(encoding='utf8',newline='') as f:readout=list(csv.DictReader(f))
assert len(readout)==384
for row in readout:
    condition='P0' if row['condition']=='B5_L_letter' else row['condition']
    g=group[row['id'],condition];k=int(row['K']);i=int(row['candidate_display_index'])
    value=bycid[g['consumers'][i]['consumer_id']][k-1]
    assert float(row['raw_sum'])==value['sum_logprob'] and float(row['char_normalized'])==value['char_normalized']
    assert int(row['raw_prediction'])==pred(g,k)[0] and int(row['char_normalized_prediction'])==pred(g,k,'char_normalized')[0]

first4=ids[:4]
cells=[('B5_L_letter','P0'),('B5_L_full','B5_L_full'),('B5_F_letter','B5_F_letter'),('B5_F_full','B5_F_full')]
for field,label in [('sum_logprob','raw'),('char_normalized','char_norm')]:
    values={}
    for alias,condition in cells:
        values[alias]={}
        for k in range(1,5):
            x=[int(pred(group[q,condition],k,field)[1]==group[q,condition]['expected_original_content_index']) for q in first4]
            values[alias][k]=np.asarray(x,float);checkstat(('B5_target_match_'+label,alias,str(k)),x,first4)
        checkstat(('B5_K4_minus_K1_target_match_'+label,alias,'1to4'),values[alias][4]-values[alias][1],first4)
    contrasts={}
    for k in range(1,5):
        a,b,c,d=[values[alias][k] for alias,condition in cells]
        contrasts[k]={'demo_effect_letter_target':c-a,'demo_effect_full_target':d-b,'target_effect_letter_demo':b-a,'target_effect_full_demo':d-c,'accuracy_factor_interaction':(d-b)-(c-a)}
        for name,x in contrasts[k].items():checkstat(('B5_'+name+'_'+label,'paired_same4_factorial',str(k)),x,first4)
    for name in contrasts[1]:checkstat(('B5_K4_minus_K1_'+name+'_'+label,'paired_same4_factorial','1to4'),contrasts[4][name]-contrasts[1][name],first4)

h1={}
def store(name,variant,k,q,value):h1.setdefault((name,variant,k),{})[q]=float(value)
for variant in ('placeholder','matched'):
    qids=[q for q in ids if (q,'Q_'+variant) in group]
    for q in qids:
        j=next(x['wrong_slot'] for x in side['cohort'] if x['id']==q)
        n=group[q,'P0']['nchoices'];keep=[i for i in range(n) if i!=j]
        pairs=list(itertools.combinations(keep,2));gold=group[q,'P0']['expected_original_content_index']
        for k in range(1,5):
            iq=I(q,variant,'Q',k);inn=I(q,variant,'N',k)
            for family,iv in [('Q',iq),('N',inn)]:
                jv=np.asarray([iv[i]-iv[l] for i,l in pairs])
                store('meanabs_J_'+family,variant,k,q,np.abs(jv).mean())
                store('RMS_J_'+family,variant,k,q,np.sqrt(np.mean(jv**2)))
                store('mean_I_'+family,variant,k,q,iv[keep].mean())
                store('gold_relative_margin_response_'+family,variant,k,q,np.mean([iv[gold]-iv[i] for i in keep if i!=gold]))
            did=iq-inn;jv=np.asarray([did[i]-did[l] for i,l in pairs])
            store('meanabs_J_QminusN',variant,k,q,np.abs(jv).mean())
            store('RMS_J_QminusN',variant,k,q,np.sqrt(np.mean(jv**2)))
            store('gold_relative_margin_response_QminusN',variant,k,q,np.mean([did[gold]-did[i] for i in keep if i!=gold]))
        for family,baseline,modified in [('Q','P0','Q_'+variant),('N','N_original','N_'+variant)]:
            a,b=group[q,baseline],group[q,modified];pa=pred(a,4)[0];pb=pred(b,4)[0]
            store('K4_display_switch_'+family,variant,'final',q,pa!=pb)
            store('K4_content_text_switch_'+family,variant,'final',q,a['choices'][pa]!=b['choices'][pb])
for (name,variant,k),values in h1.items():
    qids=[q for q in ids if q in values]
    checkstat((name,variant,str(k)),[values[q] for q in qids],qids,approximate=True)
    if isinstance(k,int) and k<4:
        nxt=h1[name,variant,k+1]
        checkstat(('adjacent_delta_'+name,variant,str(k)+'to'+str(k+1)),[nxt[q]-values[q] for q in qids],qids,approximate=True)
assert checked_stats+approximate_stats==len(statistics)==548

with (P/'native_loop_transitions.csv').open(encoding='utf8',newline='') as f:loops=list(csv.DictReader(f))
assert len(loops)==1176
for row in loops:
    g=group[row['id'],row['condition']];k=int(row['K_from']);field='sum_logprob' if row['readout']=='raw' else 'char_normalized';gold=g['expected_original_content_index']
    p,a=pred(g,k,field);p,b=pred(g,k+1,field)
    assert (row['rescue']=='True')==(a!=gold and b==gold) and (row['damage']=='True')==(a==gold and b!=gold)
    assert (row['content_changed']=='True')==(a!=b)

result=dict(status='passed_Root_independent_full_raw_payload_and_effect_replay',questions=24,
    logical_groups=196,complete_consumers=784,actual_body_calls=218,native_head_calls=872,
    vectors_verified=len(index),raw_NPZ_sha256=complete['VECTORS_sha256'],
    native_rows_exact=len(native),permutation_rows_exact=len(permutation),distractor_I_J_DID_rows_exact=len(effects),
    question_level_point_and_bootstrap_rows_independently_exact=checked_stats,
    additional_H1_stat_rows_within_1e_12_cross_architecture_tolerance=approximate_stats,
    all548_statistics_replayed=True,max_point_or_CI_absolute_replay_difference=max_stat_error,
    content_letter_readout_rows_exact=len(readout),adjacent_loop_transition_rows_exact=len(loops),
    raw_target_match_counts_K1_to_K4=counts,fullV_FP64_normalizer_max_abs_independent_error=max_norm_error,
    native_FP32_token_LP_vs_FP64_reference_max_abs_error=max_lp_reference_error,
    reference_errors_are_sanity_only_actual_GPU_scores_used=True,parameters_unchanged=269,
    GPU_worker_seconds=budget['used_seconds'],original_21_ledger_total_seconds=budget['total_used_seconds'],
    remaining_original_budget_seconds=budget['remaining_seconds'],no_new_model_or_GPU_run=True,
    scope='Independent arithmetic replay of the complete pilot, not independent new benchmark evidence or an internal causal interpretation.')
(P/'ROOT_POSTRUN_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
