"""Equal-question paired response analysis, no gold or new answer metric."""
import csv,itertools
import numpy as np
from r17_common import ROOT,LOCK,read,sha,digest,jsonlines,save,gate


def bootstrap(values):
    x=np.asarray(values,dtype=float);assert x.shape==(12,);rng=np.random.default_rng(2021)
    means=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(-1)
    return dict(mean=float(x.mean()),median=float(np.median(x)),question_bootstrap95_CI=np.quantile(means,[.025,.975]).tolist(),question_units=12,bootstrap_draws=10000)


def exact_paired_tail(values):
    x=np.asarray(values);observed=x.sum();mass=[float((x*np.asarray(signs)).sum()) for signs in itertools.product([-1,1],repeat=len(x))]
    return dict(mean_advantage=float(x.mean()),exact_one_sided_sign_flip_p=sum(v>=observed for v in mass)/len(mass),assignments=len(mass),scope='Exploratory paired12-question symmetry reference, not a test of arbitrary causal semantic transport.')


def main():
    spec=gate();done=read('results/r17_paired_COMPLETE.json');sup=read('logs/r17_SUPERVISOR.json')
    assert done['status']==sup['status']=='complete' and done['n']==12 and not done['weights_modified'] and not done['heldout_accessed']
    assert sha(done['score_file'])==done['score_file_sha256'] and done['protocol_sha256']==sha(LOCK) and done['head_norm_before']==done['head_norm_after']
    records=list(jsonlines(done['score_file']));assert [r['id'] for r in records]==[r['id'] for r in spec['cohort']]
    names=['reference','transfer','reference_shuffle','transfer_shuffle','reference_native_metric','static'];questions=[];flat=[];complete=True;positions=0
    for r,c in zip(records,spec['cohort']):
        assert r['signature']==done['signature']==digest(dict(protocol_sha256=sha(LOCK),cohort=spec['cohort'])) and r['cohort_entry_sha256']==digest(c) and r['main_reference_parity_error']==0.
        m={k:np.asarray(v) for k,v in r['metrics'].items()};assert all(len(v)==r['tokens'] for v in m.values());positions+=r['tokens'];q=dict(id=r['id'],split=r['split'],tokens=r['tokens'])
        for name in names:
            valid=m[name+'_defined'].astype(bool);q[name+'_defined_positions']=int(valid.sum())
            for k in ['cosine','R2']:q[name+'_'+k+'_mean']=float(m[name+'_'+k][valid].mean()) if valid.any() else None
            if name in names[:4]:complete&=bool(valid.all())
        valid=m['identity_error_defined'].astype(bool);q['identity_error_mean']=float(m['identity_error'][valid].mean()) if valid.any() else None
        q['reference_response_energy_fraction']=float(np.divide(m['reference_b_norm2'],m['reference_a_norm2'],out=np.zeros(r['tokens']),where=m['reference_a_norm2']>0).mean())
        q['real_response_energy_fraction']=float(np.divide(m['transfer_b_norm2'],m['static_b_norm2'],out=np.zeros(r['tokens']),where=m['static_b_norm2']>0).mean())
        questions.append(q)
        for t in range(r['tokens']):flat.append(dict(id=r['id'],position=t,**{k:(bool(v[t]) if v.dtype==bool else float(v[t])) for k,v in m.items()}))
    summaries={};paired={}
    if complete:
        for key in questions[0]:
            if key.endswith('_mean') or key.endswith('_fraction'):
                values=[q[key] for q in questions]
                summaries[key]=bootstrap(values) if all(v is not None for v in values) else dict(status='undefined_auxiliary_summary',questions_with_undefined=sum(v is None for v in values))
        for name in ['reference','transfer']:
            difference=[q[name+'_R2_mean']-q[name+'_shuffle_R2_mean'] for q in questions];paired[name]=exact_paired_tail(difference)
        relevant=summaries['reference_R2_mean']['question_bootstrap95_CI'][0]>.5 and paired['reference']['exact_one_sided_sign_flip_p']<.05
        low_transfer=summaries['transfer_R2_mean']['question_bootstrap95_CI'][1]<.5
        high_transfer=summaries['transfer_R2_mean']['question_bootstrap95_CI'][0]>.5 and summaries['transfer_cosine_mean']['question_bootstrap95_CI'][0]>0
        decision='supports_low_mean_prefix_projection_for_this_cue' if relevant and low_transfer else 'high_mean_prefix_reference_and_positive_transfer_for_this_cue' if relevant and high_transfer else 'inconclusive_for_reference_direction_transport'
    else:relevant=low_transfer=high_transfer=False;decision='undefined_direction_positions_block_operational_screen'
    for path,data in [('results/r17_question_responses.csv',questions),('results/r17_prefix_responses.csv',flat)]:
        with (ROOT/path).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    result=dict(status='complete_fixed_cue_observation',n=12,positions=positions,protocol_sha256=sha(LOCK),all_primary_and_shuffle_positions_defined=complete,
        question_summaries=summaries,paired_shuffle_comparisons=paired,operational_screen=dict(reference_mean_prefix_R2_supported=relevant,transfer_mean_prefix_R2_low=low_transfer,
            transfer_mean_prefix_R2_positive_supported=high_transfer,decision=decision,threshold_R2=.5,interpretation='Equal-question mean of per-prefix projection fractions, not pooled response energy. Bootstrap interval is exploratory with12 questions; not a decoder or universal transport test.'),
        new_method_candidates=0,new_accuracy_results=False,heldout_accessed=False,external_benchmarks_started=False,automatic_extension_authorized=False,
        limitations='Answer->Response affects only the current answer cue, not demonstrations/question. It is not known pure nuisance. Reference-direction irrelevance can only refer to this specific cue; a negative transfer result is not a general semantic theorem. No outcome-based sample expansion or alternate wording search.')
    save('results/r17_RESPONSE_ANALYSIS.json',result);print(result['status'],decision,'positions',positions,flush=True)


if __name__=='__main__':main()
