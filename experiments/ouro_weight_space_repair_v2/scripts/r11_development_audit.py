"""Posthoc development-only algebra and cyclic-reference stability diagnostics.

No new eligible recipe, held-out row, model call or selection change. Attribution
terms explain score arithmetic; they do not establish hidden-state semantics.
"""
import csv,json,time
import numpy as np
from r7_core import ROOT,read,sha,jsonlines
from r11_scores import all_scores,ELIGIBLE
from slcd_r6_scores import _inputs,projection_parameters


def csvout(path,values):
    with (ROOT/path).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in values for k in r)));w.writeheader();w.writerows(values)


def main():
    start=time.time();protocol=read('configs/r11_PROTOCOL.json');d=read('results/r11_DEVELOPMENT_COMPLETE.json')
    assert d['n']==576 and d['protocol_sha256']==sha('configs/r11_PROTOCOL.json') and sha(d['score_file'])==d['score_file_sha256']
    assert {n:sha('scripts/'+n) for n in protocol['source_sha256']}==protocol['source_sha256']
    ar=read('results/r10_development_initial_COMPLETE.json');assert sha('results/r10_development_initial_COMPLETE.json')==d['r10_auxiliary_receipt_sha256'] and sha(ar['score_file'])==ar['score_file_sha256']
    data=list(jsonlines(d['score_file']));aux={r['id']:r for r in jsonlines(ar['score_file'])};refs={}
    for split in ['discovery','selection']:
        path='results/r7_'+split+'_development.jsonl';assert sha(path)==protocol['r7_score_file_sha256'][split]
        refs.update({r['id']:r for r in jsonlines(path)})
    assert {r['id'] for r in data}==set(aux)==set(refs)
    decomposition=[];stability=[];maxerror=0.;max_cyclic_error=0.
    for row in data:
        ref=refs[row['id']];q=aux[row['id']];gold=row['gold'];assert ref['gold']==q['gold']==gold
        original,prior=ref['original_methods'],ref['prior_methods'][0];a,g,e=_inputs(original,prior);old=np.asarray(ref['baseline_scores']['M0_R6']);oldalpha=projection_parameters(g,e)[0]
        replay,_=all_scores(original,prior,q)
        for k,v in replay.items():assert np.array_equal(v,row['method_scores'][k])
        rivals=[j for j in range(len(a)) if j!=gold];rival=max(rivals,key=lambda j:old[j]);oldcorrect=int(old.argmax())==gold
        for name in ELIGIBLE:
            diag=row['diagnostics'][name];score=np.asarray(row['method_scores'][name]);alpha=diag.get('alpha',oldalpha);protected=np.asarray(diag.get('removed_component',[0.]*len(a)))
            coefficient=(oldalpha-alpha)*e;direction=alpha*protected;delta=score-old;residual=delta-coefficient-direction;maxerror=max(maxerror,float(np.max(np.abs(residual))))
            # The two changes interact. Average both orderings (two-factor
            # Shapley arithmetic) rather than treating one ordering as unique.
            average_direction=.5*(oldalpha+alpha)*protected
            average_coefficient=(oldalpha-alpha)*(e-.5*protected)
            assert np.max(np.abs(delta-average_direction-average_coefficient))<1e-12
            robust=name=='robust_cyclic_loop_axis';rr=q['cyclic_rotations'] if robust else q['cyclic_rotations'][:1]
            j1=np.mean([r['sum_scores']['native_k1'] for r in rr],axis=0);j4=np.mean([r['sum_scores']['native_k4'] for r in rr],axis=0)
            transition=('Correct' if oldcorrect else 'Wrong')+'->'+('Correct' if int(score.argmax())==gold else 'Wrong')
            margin=lambda v:float(v[gold]-v[rival])
            decomposition.append(dict(id=row['id'],method=name,transition=transition,gold=gold,fixed_old_R6_rival=rival,
                old_alpha=oldalpha,new_alpha=alpha,alpha_increased=alpha>oldalpha,old_zero_new_active=oldalpha==0 and alpha>0,
                old_correction_norm=float(np.linalg.norm((old-a)-(old-a).mean())),new_correction_norm=float(np.linalg.norm((score-a)-(score-a).mean())),
                old_fixed_rival_margin=margin(old),new_fixed_rival_margin=margin(score),actual_fixed_rival_margin_change=margin(delta),
                coefficient_first_margin_change=margin(coefficient),protection_second_margin_change=margin(direction),
                averaged_coefficient_margin_change=margin(average_coefficient),averaged_protection_margin_change=margin(average_direction),
                interaction_margin=margin((alpha-oldalpha)*protected),algebra_max_abs_residual=float(np.max(np.abs(residual))),
                reference_consensus=diag['protection_consensus'],auxiliary_loop1_correct=int(j1.argmax())==gold,auxiliary_loop4_correct=int(j4.argmax())==gold,
                auxiliary_loop_rescue=int(j1.argmax())!=gold and int(j4.argmax())==gold))
        # A cyclic permutation reuses exactly the same n saved model contexts,
        # now shifted and mapped to the new candidate indices. No synthetic
        # vocabulary logits or unseen-reference predictions are fabricated.
        n=len(a)
        for shift in range(1,n):
            order=[(j+shift)%n for j in range(n)]
            permute=lambda methods:{k:{**v,'normalized_scores':np.asarray(v['normalized_scores'])[order].tolist(),
                                        'sum_scores':np.asarray(v['sum_scores'])[order].tolist()} for k,v in methods.items()}
            mapped=[]
            for offset in range(n):
                source=q['cyclic_rotations'][(shift+offset)%n]
                mapped.append(dict(sum_scores={k:np.asarray(v)[order].tolist() for k,v in source['sum_scores'].items()}))
            newaux=dict(auxiliary_sum_scores=mapped[0]['sum_scores'],cyclic_rotations=mapped)
            result,_=all_scores(permute(original),permute(prior),newaux)
            for name in ELIGIBLE:
                expected=np.asarray(row['method_scores'][name])[order];got=result[name];error=float(np.max(np.abs(got-expected)))
                if name=='robust_cyclic_loop_axis':max_cyclic_error=max(max_cyclic_error,error);assert error<1e-12
                new_original_index=order[int(got.argmax())];base_original_index=int(np.argmax(row['method_scores'][name]))
                stability.append(dict(id=row['id'],method=name,cyclic_shift=shift,max_mapped_score_change=error,
                    prediction_changed=new_original_index!=base_original_index,context_source='Exact same complete saved cyclic model forwards; mapped to new indices'))
    assert maxerror<1e-12
    csvout('results/r11_development_margin_decomposition.csv',decomposition);csvout('results/r11_development_cyclic_stability.csv',stability)
    summary=[]
    for method in ELIGIBLE:
        for transition in ['all','Wrong->Correct','Correct->Wrong','Correct->Correct','Wrong->Wrong']:
            group=[r for r in decomposition if r['method']==method and (transition=='all' or r['transition']==transition)]
            if not group:continue
            summary.append(dict(method=method,transition=transition,n=len(group),old_zero_new_active=sum(r['old_zero_new_active'] for r in group),
                alpha_increased=sum(r['alpha_increased'] for r in group),correction_norm_increased=sum(r['new_correction_norm']>r['old_correction_norm'] for r in group),
                positive_averaged_protection_margin=sum(r['averaged_protection_margin_change']>0 for r in group),
                positive_averaged_coefficient_margin=sum(r['averaged_coefficient_margin_change']>0 for r in group),
                auxiliary_loop_rescues=sum(r['auxiliary_loop_rescue'] for r in group),
                mean_protection_margin=float(np.mean([r['averaged_protection_margin_change'] for r in group])),
                mean_coefficient_margin=float(np.mean([r['averaged_coefficient_margin_change'] for r in group])),
                mean_reference_consensus=float(np.mean([r['reference_consensus'] for r in group]))))
    csvout('results/r11_development_mechanism_summary.csv',summary)
    result=dict(status='passed_posthoc_development_diagnostics',n=576,source_sha256=sha('scripts/r11_development_audit.py'),
        protocol_sha256=sha('configs/r11_PROTOCOL.json'),max_score_algebra_error=maxerror,max_robust_cyclic_score_error=max_cyclic_error,
        summary=summary,stability={m:dict(perturbations=sum(r['method']==m for r in stability),prediction_changes=sum(r['prediction_changed'] for r in stability if r['method']==m)) for m in ELIGIBLE},
        additional_GPU_seconds=0,elapsed_seconds=time.time()-start,
        limits='Posthoc development only; no inference formula/selection change. Score algebra is order-dependent with interactions; averaged terms are accounting, not a unique semantic or hidden-state causal explanation. Robust invariance is cyclic, not arbitrary n-factorial option permutation.')
    savepath=ROOT/'results/r11_DEVELOPMENT_MECHANISM_AUDIT.json';savepath.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
